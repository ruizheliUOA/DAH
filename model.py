import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
randn = lambda *x: torch.randn(*x)

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

class CRF(nn.Module):
    def __init__(self, num_tags, device):
        super(CRF, self).__init__()
        self.batch_size = 0
        self.num_tags = num_tags
        self.device = device
        # self.hidden_dim = hidden_dim
        # self.hidden2tag = nn.Linear(hidden_dim*2, num_tags)

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags)).to(self.device)
        self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def forward(self, h, mask): # forward algorithm
        # gru_features: [batch, max_conv_len, hidden * num_direction]
        # h = self.hidden2tag(gru_features)
        # h *= mask.unsqueeze(2)
        # initialize forward variables in log space
        score = torch.FloatTensor(self.batch_size, self.num_tags).fill_(-10000).to(self.device) # [B, C]
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score # partition function

    def score(self, h, y0, mask): # calculate the score of a given sequence
        score = torch.FloatTensor(self.batch_size).fill_(0.).to(self.device)
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y0[t + 1]] for h, y0 in zip(h, y0)])
            trans_t = torch.cat([trans[y0[t + 1], y0[t]] for y0 in y0])
            score += (emit_t + trans_t) * mask_t
        last_tag = y0.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = torch.LongTensor().to(self.device)
        score = torch.FloatTensor(self.batch_size, self.num_tags).fill_(-10000).to(self.device)
        score[:, SOS_IDX] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b] # best tag
            j = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:j]):
                i = bptr_t[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()
    
        return best_path

def create_emb_layer(weights_matrix,num_embeddings, embedding_dim, freeze_embed=False):
    # num_embeddings, embedding_dim = weights_matrix.size()
    # print('initial this function')

    emb_layer = nn.Embedding.from_pretrained(weights_matrix, freeze=freeze_embed)
    # emb_layer.load_state_dict({'weight': weights_matrix})
    

    return emb_layer

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, weights_matrix,
                 n_layers=1, dropout=0.5, freeze_embed=False):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.weights_matrix = weights_matrix
        
        self.embed= create_emb_layer(self.weights_matrix, self.vocab_size, self.embed_size, freeze_embed=freeze_embed)
        # self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, hidden=None):
        # src: [max_sent_len, batch]
        # embedded: (max_sent_len, batch, embed_size)
        
        embedded = self.dropout(self.embed(src))
        # output: all hidden state from t==0 to t==token_len (token_len, batch, hidden_size * num_direction)
        # hidden: the hidden state at t==token_len [num_layers * num_direction, batch, hidden_size]
        outputs, hidden = self.gru(embedded, hidden)
        
        # sum bidirectional outputs
        # outputs = (outputs[:, :, :self.hidden_size] +
        #            outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # input dimension is hidden_size * 3
        self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, encoder_outputs, last_time_hidden, multi_de_last_time_hidden):
        # act_conv_hidden: [num_layer, batch, hidden_size]
        # topic_conv_hidden: [num_layer, batch, hidden_size]
        # encoder_outputs: [max_sent_len, batch, hidden_size]

        timestep = encoder_outputs.shape[0]
        # num_decoder_layer = hidden.size(0)

        if last_time_hidden.shape[0] == 2:
            last_time_hidden = (last_time_hidden[:1,:,:] + last_time_hidden[1:,:,:])
            multi_de_last_time_hidden = (multi_de_last_time_hidden[:1,:,:] + multi_de_last_time_hidden[1:,:,:])

        # act_h: [batch, max_sent_len, hidden_size]
        h = last_time_hidden.repeat(timestep, 1, 1).transpose(0, 1)
        a_h = multi_de_last_time_hidden.repeat(timestep, 1, 1).transpose(0, 1)

        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch, max_sent_len, hidden_size]
        attn_energies = self.score(h, encoder_outputs, a_h)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, last_time_hidden, encoder_outputs, multi_de_last_time_hidden):
        # [B*T*3H]->[B*T*H]
        # the original code here has a bug, it ignores the nolinear function tanh

        energy = torch.tanh(self.attn(torch.cat((last_time_hidden, encoder_outputs, multi_de_last_time_hidden), 2))) # [B, T, H]
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        # self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,
                          n_layers, dropout=dropout)
        
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, encoder_outputs, last_time_hidden, multi_de_last_time_hidden):
        # encoder_outputs: [max_sent_len, batch, hidden_size] (last layer)
        # last_time_hidden: [num_layer,batch,hidden]
        # multi_de_last_time_hidden: [num_layer,batch,hidden]




        # Get the embedding of the current input word (last output word)
        # embedded = self.embed(input).unsqueeze(0)  # (1,Batch_size,embeded_size)
        # embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        # encoder_outputs: [token_len, batch, hidden_size]
        # last_hidden: (decoder_layer, batch, hidden_size)
        # attn_weight: (batch, 1, seq_len)

        # last_hidden[-1] has problem
        attn_weights = self.attention(encoder_outputs, last_time_hidden, multi_de_last_time_hidden) #[B,1,T]
        
        att_context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,hidden_size)
        
        att_context = att_context.transpose(0, 1)  # (1,B,hidden_size)
        # Combine embedded input word and attended context, run through RNN
        
        # rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(att_context, last_time_hidden)
        # output: [1,batch,hidden] last layer
        # hidden: [num_layer,B,hidden]

        # output = output.squeeze(0)  # (1,B,hidden_size) -> (B,hidden_size)
        # context = context.squeeze(0)

        # output = self.out(torch.cat([output, context], 1))
        # # output = softmax(linear(hidden_state,attention))
        # output = F.log_softmax(output, dim=1)
        return output.squeeze(0), hidden




class Seq2Seq(nn.Module):
    def __init__(self, encoder, act_l_decoder, act_r_decoder, topic_l_decoder, topic_r_decoder, act_CRF, topic_CRF, hidden_dim, act_tag_size, topic_tag_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.act_l_decoder = act_l_decoder
        self.act_r_decoder = act_r_decoder
        self.topic_l_decoder = topic_l_decoder
        self.topic_r_decoder = topic_r_decoder
        self.act_CRF = act_CRF
        self.topic_CRF = topic_CRF
        self.device = device
        self.hidden_dim = hidden_dim
        self.act_tag_size = act_tag_size
        self.topic_tag_size = topic_tag_size
        self.hidden2act_tag = nn.Linear(hidden_dim*2, act_tag_size)
        self.hidden2topic_tag = nn.Linear(hidden_dim*2, topic_tag_size)
        

    def forward(self, convs, act_trg, topic_trg):
        # convs: (max_conv_len, batch, max_sent_len) max_len means the biggest length of utterance in conversation
        # trg: should be [max_conv_len, batch]
        
        batch_size = convs.shape[1]
        hidden_dim = self.act_l_decoder.hidden_size
        self.act_CRF.batch_size = batch_size
        self.topic_CRF.batch_size = batch_size
        num_layers = self.act_l_decoder.n_layers
        
        max_conv_len = convs.shape[0]

        # act_mast: [max_conv_len-1, batch]
        act_mask = act_trg[1:, :].gt(PAD_IDX).float()
        topic_mask = topic_trg[1:, :].gt(PAD_IDX).float()

        

        # act_size = self.act_decoder.output_size
        # topic_size = self.topic_decoder.output_size

        # act_outputs = torch.zeros((max_conv_len, batch_size, act_size), device=self.device)
        # topic_outputs = torch.zeros((max_conv_len, batch_size, topic_size), device=self.device)

        
        # act_output = torch.tensor([2]*batch_size, device=self.device)
        # topic_output = torch.tensor([2]*batch_size, device=self.device)

        encoder_bi_outputs = []
    

        for t in range(max_conv_len):
            encoder_output, _ = self.encoder(convs[t,:,:].transpose(0,1))
            encoder_bi_outputs.append(encoder_output)
        
        # encoder_bi_outputs: [max_conv_len]
        # inside: [max_sent_len, batch, hidden_size * num_direction] (last layer)

        act_l_ = []
        act_r_ = []
        topic_l_ = []
        topic_r_ = []

        for t in range(max_conv_len):
            # act_gru_l_features should be [batch,hidden]

            if t == 0:
                # act_l_decoder(all_left_direction_utt_hidden, left_direction_act_conv_gru_hidden, left_direction_topic_conv_gru_hidden)
                act_gru_l_features, act_l_hidden = self.act_l_decoder(encoder_bi_outputs[t][:,:,:hidden_dim], encoder_bi_outputs[t][-1,:,:hidden_dim].repeat(num_layers,1,1),encoder_bi_outputs[t][-1,:,:hidden_dim].repeat(num_layers,1,1))
                topic_gru_l_features, topic_l_hidden = self.topic_l_decoder(encoder_bi_outputs[t][:,:,:hidden_dim], encoder_bi_outputs[t][-1,:,:hidden_dim].repeat(num_layers,1,1),encoder_bi_outputs[t][-1,:,:hidden_dim].repeat(num_layers,1,1))

                act_gru_r_features, act_r_hidden = self.act_r_decoder(encoder_bi_outputs[t][:,:,hidden_dim:], encoder_bi_outputs[t][0,:,hidden_dim:].repeat(num_layers,1,1),encoder_bi_outputs[t][0,:,hidden_dim:].repeat(num_layers,1,1))
                topic_gru_r_features, topic_r_hidden = self.topic_r_decoder(encoder_bi_outputs[t][:,:,hidden_dim:], encoder_bi_outputs[t][0,:,hidden_dim:].repeat(num_layers,1,1),encoder_bi_outputs[t][0,:,hidden_dim:].repeat(num_layers,1,1))
            
            else:

                act_gru_l_features, act_l_hidden = self.act_l_decoder(encoder_bi_outputs[t][:,:,:hidden_dim], act_l_hidden, topic_l_hidden)
                topic_gru_l_features, topic_l_hidden = self.topic_l_decoder(encoder_bi_outputs[t][:,:,:hidden_dim], topic_l_hidden, act_l_hidden)

                act_gru_r_features, act_r_hidden = self.act_r_decoder(encoder_bi_outputs[t][:,:,hidden_dim:], act_r_hidden, topic_r_hidden)
                topic_gru_r_features, topic_r_hidden = self.topic_r_decoder(encoder_bi_outputs[t][:,:,hidden_dim:], topic_r_hidden, act_r_hidden)

            # 
            act_l_.append(act_gru_l_features)
            act_r_.append(act_gru_r_features)
            topic_l_.append(topic_gru_l_features)
            topic_r_.append(topic_gru_r_features)

        # conv_act_l: [max_conv_len, batch, hidden]
        conv_act_l = torch.stack(act_l_)
        conv_act_r = torch.stack(act_r_)
        conv_topic_l = torch.stack(topic_l_)
        conv_topic_r = torch.stack(topic_r_)

        # act_geu_features: [max_conv_len, batch, hidden*2]
        act_gru_features = torch.cat((conv_act_l,conv_act_r), dim=2)
        topic_gru_features = torch.cat((conv_topic_l,conv_topic_r), dim=2)

        act_h = self.hidden2act_tag(act_gru_features.transpose(0,1))
        topic_h = self.hidden2topic_tag(topic_gru_features.transpose(0,1))

        
        # act_gru_features: [max_conv_len, batch, hidden * num_direction]
        if self.training:
            act_scores = self.act_CRF.forward(act_h, act_mask.transpose(0,1))
            topic_scores = self.topic_CRF.forward(topic_h, topic_mask.transpose(0,1))

        
            act_gold_scores = self.act_CRF.score(act_h, act_trg.transpose(0,1), act_mask.transpose(0,1))
            topic_gold_scores = self.topic_CRF.score(topic_h, topic_trg.transpose(0,1), topic_mask.transpose(0,1))

            return torch.mean(act_scores - act_gold_scores), torch.mean(topic_scores - topic_gold_scores) # NLL loss
        else:
            act_scores = self.act_CRF.forward(act_h, act_mask.transpose(0,1))
            topic_scores = self.topic_CRF.forward(topic_h, topic_mask.transpose(0,1))

        
            act_gold_scores = self.act_CRF.score(act_h, act_trg.transpose(0,1), act_mask.transpose(0,1))
            topic_gold_scores = self.topic_CRF.score(topic_h, topic_trg.transpose(0,1), topic_mask.transpose(0,1))

            act_best_path = self.act_CRF.decode(act_h, act_mask.transpose(0,1))
            topic_best_path = self.topic_CRF.decode(topic_h, topic_mask.transpose(0,1))

            return act_best_path, topic_best_path, torch.mean(act_scores - act_gold_scores), torch.mean(topic_scores - topic_gold_scores)

        



        

        
