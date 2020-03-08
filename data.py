import os
import re
import random
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset
import operator
from itertools import groupby
from functools import reduce
import torch
import pickle
import csv
# import os

class Corpus(Dataset):
    def __init__(self, conv_path, conv_topic_path, da_path, word_dic=None, da_id=None, topic_id=None, min_word_count=1):
        print("start to load Corpus data")

        file_list = []
        for path, subdirs, files in os.walk(conv_path):
            for single_file in files:
                if 'DS_Store' in single_file:
                    continue
                file_list.append(conv_path+single_file)


        self.convs = []
        self.conv_id = []
        self.convs_token = []
        # self.sents = []

        for file_path in file_list:
            with open(file_path, 'r') as f:
                conv = f.readlines()

            self.conv_id.append(file_path[-12:-8])
            self.sents = [self._split(i.strip()) for i in conv]
            # 长短排序
            # conv = [(t, len(t)) for t in conv]
            # corpus.sort(key=operator.itemgetter(1))
            # self.conv = [x for x, _ in conv]
            self.convs.append(self.sents)
        self.convs = zip(self.convs,self.conv_id)
        convs_list = [(t,len(t), conv_id) for t, conv_id in self.convs]
        convs_list.sort(key=operator.itemgetter(1))
        self.convs = [(x, conv_id) for x, _, conv_id in convs_list]
            
        print("start to build dictionary")
        if word_dic is not None:
            self.word_id = word_dic
        else:
            doc = " ".join([" ".join(sent_) for conv_, _ in self.convs for sent_ in conv_])
            self.word_id = self._make_dic(doc, min_word_count)
        self.id_word = self._make_inv_dic(self.word_id)
        self.voca_size = len(self.word_id)
        self.sos_token = torch.tensor([2])
        self.eos_token = torch.tensor([3])
        print("start to make one-hot vectors")
        for conv_, conv_id in self.convs:
            self.textcodes = [self._txt_vecs(sent_) for sent_ in conv_]
            self.convs_token.append([self.textcodes, conv_id])
        
        print("start to load da data")

        da_file_list = []
        for path, subdirs, files in os.walk(da_path):
            for single_file in files:
                if 'DS_Store' in single_file:
                    continue
                da_file_list.append(da_path+single_file)

        da_conv_id = []
        
        self.convs_da_tags = []
        for file_path in da_file_list:
            new_da_tags = []
            with open(file_path,'r') as f:
                da_tags = f.readlines()

            da_conv_id.append(file_path[-12:-8])

            for da_tag_ in da_tags:
                _, true_da, _, _, _ = da_tag_.strip('\n').split('\t')
                new_da_tags.append(true_da)
            self.convs_da_tags.append(new_da_tags)   

        # convs_list = [(t,len(t), conv_id) for t, conv_id in self.convs]
        # convs_list.sort(key=operator.itemgetter(1))
        # self.convs = [(x, conv_id) for x, _, conv_id in convs_list]

        self.convs_da_tags = zip(self.convs_da_tags,da_conv_id)
        convs_da_list = [(t,len(t), conv_id) for t, conv_id in self.convs_da_tags]
        convs_da_list.sort(key=operator.itemgetter(1))
        self.convs_da_tags = [(x, conv_id) for x, _, conv_id in convs_da_list]

        print("start to build DA dictionary")

        if da_id is not None:
            self.da_id = da_id
        else:
            doc = " ".join([" ".join(conv_) for conv_, _ in self.convs_da_tags])
            self.da_id = self._make_da_dic(doc)
        self.id_da = self._make_inv_dic(self.da_id)
        # self.da_voca_size = len(self.da_id)
        self.da_size = len(self.da_id)
        # print(self.da_id)

        print("start to make one-hot vectors")
        self.new_convs_token = self.convs_token[:]
        for das_, conv_id in self.convs_da_tags:
            self.da_codes = [self._da_vecs(sent_da_) for sent_da_ in das_]
            self.da_codes = [torch.tensor(1)]+self.da_codes

            for i, conv_token in enumerate(self.convs_token):
                if conv_id == conv_token[1]:
                    self.new_convs_token[i].append(self.da_codes)
                    break
            
        
        print("start to load topic data")
        with open(conv_topic_path,'r') as f:
            content = f.readlines()

        topic_conv_id = []
        self.topic_list = []

        for line in content:
            topic_id_, topic_cat = line.strip().split('\t') #id_num and topic name
            topic_conv_id.append(topic_id_)
            self.topic_list.append(topic_cat)
        
        print("start to build topic dictionary")

        if topic_id is not None:
            self.topic_id = topic_id
        else:
            doc = "\t".join(self.topic_list)
            self.topic_id = self._make_topic_dic(doc)
        self.id_topic = self._make_inv_dic(self.topic_id)
        # self.topic_voca_size = len(self.topic_id)
        self.topic_size = len(self.topic_id)
        # print(self.topic_id)

        print("start to make one-hot vectors")
        for index, conv_tp in enumerate(self.topic_list):
            self.tp_codes = [self._topic_vecs(conv_tp)]

            for i, conv_token in enumerate(self.convs_token):
                if topic_conv_id[index] == conv_token[1]:
                    self.new_convs_token[i].append([torch.tensor(1)]+self.tp_codes*len(conv_token[0]))
                    break

            # self.convs_token.append([self.textcodes, conv_id])
        # self.new_convs_token: [lens of convs]
        # inside: conv_token, conv_id, conv_da, conv_topic
        # conv_token: [lens of conv]; each utt: [lens of utt]
        # conv_da: [lens of conv]
        # conv_topic: [lens of conv]
        # batch: self.new_convs_token[batch_size]
        # padding: self.new_convs_token[batch_size][conv_index][0][each_utt]
        # padding: max_utt_len: len(self.new_convs_token[batch_size][conv_index][0][each_utt])
        # padding: max_conv_len: len(self.new_convs_token[batch_size][conv_index][0])
        # padding: da and topic to max_conv_len
        self.doc_size = len(self.new_convs_token) 
        # self.doc_size = len(self.sents)
        # self.max_length = max((len(i) for i in self.sents))


    def _topic_vecs(self, das):
        v = self.topic_id.get(das, 10000)
        v = torch.tensor(v)
        return v


    def _make_topic_dic(self,doc,addflag=True):
        flag_count = 3 if addflag else 1
        doc_ = re.split(r"\t", "".join(doc))
        words = sorted(doc_)
        word_count = [(w, sum(1 for _ in c)) for w, c in groupby(words)]
        word_count = [(w, c) for w, c in word_count]
        word_count.sort(key=operator.itemgetter(1), reverse=True)
        word_id = dict([(w, i+flag_count)
                        for i, (w, _) in enumerate(word_count)])
        if addflag:
            word_id["<pad>"] = 0
            # word_id["<unk0>"] = 1
            word_id["<sos>"] = 1
            word_id["<eos>"] = 2
            # self.pad_id = 0
        return word_id

    def _da_vecs(self, das):
        v = self.da_id.get(das, 10000)
        v = torch.tensor(v)
        return v


    def _make_da_dic(self, doc, addflag=True):
        flag_count = 3 if addflag else 1
        doc_ = re.split(r"\s", "".join(doc))
        words = sorted(doc_)
        word_count = [(w, sum(1 for _ in c)) for w, c in groupby(words)]
        word_count = [(w, c) for w, c in word_count]
        word_count.sort(key=operator.itemgetter(1), reverse=True)
        word_id = dict([(w, i+flag_count)
                        for i, (w, _) in enumerate(word_count)])
        if addflag:
            word_id["<pad>"] = 0
            # word_id["<unk0>"] = 1
            word_id["<sos>"] = 1
            word_id["<eos>"] = 2
            # self.pad_id = 0
        return word_id


    def _split(self, sen):
        sen = sen.lower()
        sen = re.sub(r"[.]+", ".", sen)
        # sen = re.sub(r"([.?!,]|'s)", r" \1", sen)
        return re.split(r"\s+", sen)

    def _split2(self, labels):
        labels = re.sub(r"[ ]", "-", labels)
        return re.split(r",-*", labels)

    def _make_dic(self, doc, min_word_count, addflag=True):
        flag_count = 4 if addflag else 0
        doc_ = re.split(r"\s", "".join(doc))
        words = sorted(doc_)
        word_count = [(w, sum(1 for _ in c)) for w, c in groupby(words)]
        word_count = [(w, c) for w, c in word_count if c >= min_word_count]
        word_count.sort(key=operator.itemgetter(1), reverse=True)
        word_id = dict([(w, i+flag_count)
                        for i, (w, _) in enumerate(word_count)])
        if addflag:
            word_id["<pad>"] = 0
            word_id["<unk0>"] = 1
            word_id["<sos>"] = 2
            word_id["<eos>"] = 3
            self.pad_id = 0
        return word_id

    def _make_inv_dic(self, word_id_dic):
        id_word = dict([(i, w) for w, i in word_id_dic.items()])
        return id_word

    def _word_onehot(self, word):
        v = torch.zeros([self.voca_size], dtype=torch.long)
        v[self.word_id.get(word, 1)] = 1
        return v

    def _txt_vecs(self, txt):
        v = [self.word_id.get(w, 1) for w in txt]
        v = [2]+v+[3]
        v = torch.tensor(v)
        return v

    def __getitem__(self, index):
        return self.new_convs_token[index]

    def __len__(self):
        return self.doc_size

    def totext(self, sen):
        text = [self.id_word[i] for i in sen]
        return " ".join(text)




    
