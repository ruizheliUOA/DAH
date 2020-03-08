import argparse
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import os
import time
# from tqdm import tqdm
from scipy.stats import shapiro, kstest
from BatchIter import BatchIter
import numpy as np
from data_utt import Corpus
from model import Encoder, Decoder, Seq2Seq, CRF
from sklearn.metrics import accuracy_score

base_path = "."
print("base_path=", base_path)

parser = argparse.ArgumentParser(description='DAH')
parser.add_argument('-bz', '--batch-size', type=int, default=2,
                    help='input batch size for training (default: 128)')
parser.add_argument('-em', '--embedding-size', type=int, default=300,
                    help='embedding size for training (default: 512)')
parser.add_argument('-hn', '--hidden-size', type=int, default=256,
                    help='hidden size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 500)')
parser.add_argument('--e_layers', type=int, default=2,
                    help='number of layers of rnns in encoder')
parser.add_argument('--d_layers', type=int, default=1,
                    help='number of layers of rnns in decoder')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout values')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha values')
parser.add_argument('--decay', type=float, default=1e-4,
                    help='alpha values')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='alpha values')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='alpha values')
parser.add_argument('--no-cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-dt','--dataset', type=str, default="swda",
                    help='Dataset name')
parser.add_argument('-mw','--min_word_count',type=int, default=2,
                    help='minimum word count')



args = parser.parse_args()

print(args)

if args.dataset == "swda":
    Train = Corpus(base_path+'/swda/test/test_conv/', base_path+'/test_topic_no.csv', base_path+'/swda/test/test_da/', min_word_count=args.min_word_count)
    Valid = Corpus(base_path+'/swda/valid/valid_conv/', base_path+'/swda/clean_swda_conv_topic.csv', base_path+'/swda/valid/valid_da/', word_dic=Train.word_id, da_id=Train.da_id, topic_id=Train.topic_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/swda/test/test_conv/', base_path+'/swda/clean_swda_conv_topic.csv', base_path+'/swda/test/test_da/', word_dic=Train.word_id, da_id=Train.da_id, topic_id=Train.topic_id, min_word_count=args.min_word_count)
elif args.dataset == "mrda":
    Train = Corpus(base_path+'/mrda_train_conv/', base_path+'/mrda_train_topic/', base_path+'/mrda_train_da/', min_word_count=args.min_word_count)
    Valid = Corpus(base_path+'/mrda_valid_conv/', base_path+'/mrda_valid_topic/', base_path+'/mrda_valid_da/', word_dic=Train.word_id, da_id=Train.da_id, topic_id=Train.topic_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/mrda_test_conv/', base_path+'/mrda_test_topic/', base_path+'/mrda_test_da/', word_dic=Train.word_id, da_id=Train.da_id, topic_id=Train.topic_id, min_word_count=args.min_word_count)
elif args.dataset == "dyda":
    Train = Corpus(base_path+'/dyda_train_conv/', base_path+'/dyda_train_topic/', base_path+'/dyda_train_da/', min_word_count=args.min_word_count)
    Valid = Corpus(base_path+'/dyda_valid_conv/', base_path+'/dyda_valid_topic/', base_path+'/dyda_valid_da/', word_dic=Train.word_id, da_id=Train.da_id, topic_id=Train.topic_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dyda_test_conv/', base_path+'/dyda_test_topic/', base_path+'/dyda_test_da/', word_dic=Train.word_id, da_id=Train.da_id, topic_id=Train.topic_id, min_word_count=args.min_word_count)

voca_dim = Train.voca_size
emb_dim = args.embedding_size
hid_dim = args.hidden_size
batch_size = args.batch_size
act_num_classes = Train.da_size
topic_num_classes = Train.topic_size


print(f"voca_dim={voca_dim}")

SEED = 999
lr = 0.001

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available()
                      and not args.no_cuda else 'cpu')

if torch.cuda.is_available():
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print('GPU: %s' % current_gpu_name)

dataloader_train = BatchIter(Train, batch_size)
dataloader_valid = BatchIter(Valid, batch_size)
dataloader_test = BatchIter(Test, batch_size)

sd = 1/np.sqrt(emb_dim)  # Standard deviation to use
weights = np.random.normal(0, scale=sd, size=[voca_dim, emb_dim])
weights = weights.astype(np.float32)



file = "./glove.840B.300d.txt"

# EXTRACT DESIRED GLOVE WORD VECTORS FROM TEXT FILE
count = 0
with open(file, encoding="utf-8", mode="r") as textFile:
    for line in textFile:
        # Separate the values from the word
        line = line.split()
        word = line[0]

        # If word is in our vocab, then update the corresponding weights
        word_index = Train.word_id.get(word, None)
        if word_index is not None and word_index < voca_dim:
            try :
                float(line[1])
            except:
                pass
            else:

                weights[word_index] = np.array(line[1:], dtype=np.float32)
                count += 1

print(count)
weights = torch.from_numpy(weights).requires_grad_(True)


print("[!] Instantiating models...")
encoder = Encoder(voca_dim, emb_dim, hid_dim, weights,
                    n_layers=args.e_layers, dropout=args.dropout, freeze_embed=True).to(device)
act_l_decoder = Decoder(emb_dim, hid_dim, act_num_classes,
                    n_layers=args.d_layers, dropout=args.dropout).to(device)
act_r_decoder = Decoder(emb_dim, hid_dim, act_num_classes,
                    n_layers=args.d_layers, dropout=args.dropout).to(device)
topic_l_decoder = Decoder(emb_dim, hid_dim, topic_num_classes,
                    n_layers=args.d_layers, dropout=args.dropout).to(device)
topic_r_decoder = Decoder(emb_dim, hid_dim, topic_num_classes,
                    n_layers=args.d_layers, dropout=args.dropout).to(device)

act_CRF = CRF(act_num_classes, device).to(device)
topic_CRF = CRF(topic_num_classes, device).to(device)
seq2seq = Seq2Seq(encoder, act_l_decoder, act_r_decoder, topic_l_decoder, topic_r_decoder, act_CRF, topic_CRF, hid_dim, act_num_classes, topic_num_classes, device).to(device)
optimizer = optim.Adam(seq2seq.parameters(),lr=args.lr, eps=args.eps, weight_decay=args.decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
print(seq2seq)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def acc_calculate(act_best_path, topic_best_path, true_act, true_topic):
    # true_act: [batch, max_conv_len]
    # act_predict = []
    # topic_predict = []

    
    act_tag_convs = [[Train.id_da[i] for i in conv] for conv in act_best_path]
    topic_tag_convs = [[Train.id_topic[i] for i in conv] for conv in topic_best_path]

    gold_act_convs = []
    for b in range(true_act.shape[0]):
        act_conv = []
        for i in range(true_act.shape[1]):
            if true_act[b,i].item() == 0:
                break
            act_conv.append(Train.id_da[true_act[b,i].item()])
            
        gold_act_convs.append(act_conv)

    gold_topic_convs = []
    for b in range(true_topic.shape[0]):
        topic_conv = []
        for i in range(true_topic.shape[1]):
            if true_topic[b,i].item() == 0:
                break
            topic_conv.append(Train.id_topic[true_topic[b,i].item()])
            
        gold_topic_convs.append(topic_conv)

    

    for i in range(len(act_tag_convs)):
        assert len(act_tag_convs[i]) == len(gold_act_convs[i])

    act_true = [tag for conv in gold_act_convs for tag in conv]
    act_pre = [tag for conv in act_tag_convs for tag in conv]

    topic_true = [tag for conv in gold_topic_convs for tag in conv]
    topic_pre = [tag for conv in topic_tag_convs for tag in conv]

    act_acc = accuracy_score(act_true, act_pre)
    topic_acc = accuracy_score(topic_true, topic_pre)

    return act_acc, topic_acc

def evaluate(corpus, ep):
    start_time = time.time()
    seq2seq.eval()
    total_epoch_loss = 0
    total_act_loss = 0
    total_topic_loss = 0
    total_act_acc = 0
    total_topic_acc = 0



    for i, (convs, das, topics) in enumerate(corpus):
        # das: [batch, max_conv_len]
        # batch_size = convs.shape[0]
        convs = convs.transpose(0,1).to(device)
        das = das.transpose(0,1).to(device)
        topics = topics.transpose(0,1).to(device)
        with torch.no_grad():


            act_best_path, topic_best_path, act_outputs_loss, topic_outputs_loss = seq2seq(convs, das, topics)
            act_outputs_acc, topic_outputs_acc = acc_calculate(act_best_path, topic_best_path, das.transpose(0,1)[:,1:], topics.transpose(0,1)[:,1:])

            total_loss = act_outputs_loss + args.alpha * topic_outputs_loss

            total_epoch_loss += total_loss
            total_act_loss += act_outputs_loss
            total_topic_loss += topic_outputs_loss

            # total_epoch_loss += total_loss
            # total_act_loss += act_outputs_loss
            # total_topic_loss += topic_outputs_loss
            total_act_acc += act_outputs_acc
            total_topic_acc += topic_outputs_acc

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(
        f"Time: {epoch_mins}m {epoch_secs}s| Eval {ep}: act_acc:{(total_act_acc/(i+1)):.04f}, topic_acc:{(total_topic_acc/(i+1)):.04f}, epoch_loss:{(total_epoch_loss/(i+1)):.04f}, act_loss:{(total_act_loss/(i+1)):.04f}, topic_loss:{(total_topic_loss/(i+1)):.04f}")
    
    return (total_act_acc/(i+1))

def train(corpus, ep):
    print('----------------------')
    start_time = time.time()
    seq2seq.train()
    total_epoch_loss = 0
    total_act_loss = 0
    total_topic_loss = 0
    # total_act_acc = 0
    # total_topic_acc = 0



    for i, (convs, das, topics) in enumerate(corpus):
        # convs: [max_conv_len, batch, max_sent_len]
        # das/topics: [max_conv_len, batch]
        convs = convs.transpose(0,1).to(device)
        das = das.transpose(0,1).to(device)
        topics = topics.transpose(0,1).to(device)
        # batch_size = convs.shape[1]
        optimizer.zero_grad()

        act_outputs_loss, topic_outputs_loss = seq2seq(convs, das, topics)
        total_loss = act_outputs_loss + args.alpha * topic_outputs_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 10, norm_type=2)
        optimizer.step()
        

        total_epoch_loss += total_loss
        total_act_loss += act_outputs_loss
        total_topic_loss += topic_outputs_loss
        # break
        # total_act_acc += act_outputs_acc
        # total_topic_acc += topic_outputs_acc

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # scheduler.step()
    print(
        f"Time: {epoch_mins}m {epoch_secs}s| Train {ep}: epoch_loss:{(total_epoch_loss/(i+1)):.04f}, act_loss:{(total_act_loss/(i+1)):.04f}, topic_loss:{(total_topic_loss/(i+1)):.04f}")






    
ep = 0

for ep in range(ep, args.epochs):

    train(dataloader_train, ep)
    eval_act_acc = evaluate(dataloader_valid, ep)
    test_act_acc = evaluate(dataloader_test, ep)
    scheduler.step(eval_act_acc)