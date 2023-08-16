import argparse
from curses import raw
import json
from textwrap import indent
import time
from matplotlib import pyplot as plt
import torch.nn as nn
import torch 
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.utils.data as Data
from torch.utils.data import DataLoader, Sampler, random_split
import numpy as np 
import pandas as pd 
import random 
import os
from collections import Counter
from pathlib import Path 
from torch.utils.data import SequentialSampler
from transformers import (RobertaConfig, RobertaModel, 
                        AdamW,
                        RobertaTokenizer,get_linear_schedule_with_warmup)
import logging
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_ids, nl_ids, label):
        self.code_ids = code_ids        # code tokenize ids
        self.nl_ids = nl_ids            # nl tokenize ids
        self.label = label

def text2vec(seqs,tokenizer):
    texttoken_id = []
    max_seq_len = 150
    # print(seqs)
    textoken = []
    for seq in seqs:
        textoken = textoken+[tokenizer.cls_token] + seq
    tokens_ids = tokenizer.convert_tokens_to_ids(textoken)
    texttoken_id = tokens_ids[:max_seq_len]
    texttoken_id.extend([0 for _ in range(max_seq_len -len(texttoken_id))])
    return texttoken_id

def code2vec(codes,tokenizer):
    codetoken_id = []
    max_code_len=150
    codes = eval(codes)
    code_tokens= []
    for code in codes:
        code_tokens = [tokenizer.cls_token] + code[:80]
    tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)   
    codetoken_id = tokens_ids[:max_code_len]
    codetoken_id.extend([0 for _ in range(max_code_len -len(codetoken_id))])
    return codetoken_id


        
def make_data(df,tokenizer):
    # input_batch, output_batch, target_batch = [],[],[]
    code_batch1, text_batch1,tg_batch1= [], [], []
    logging.info("Loaded the file")
    examples = []
    for index, row in df.iterrows():
        text = text2vec(eval(row['summary_processed'])+eval(row['description_processed'])+eval(row['message_processed']),tokenizer)
        code = code2vec(row['codelist_processed'],tokenizer)
        label = float(row['target'])
        code_batch1.append(code)
        text_batch1.append(text)
        tg_batch1.append(label)
        examples.append(InputFeatures(code, text, label))
    print(len(code_batch1[0]))
    print(len(tg_batch1))
    return examples
    # return torch.LongTensor(code_batch1), torch.LongTensor(text_batch1),torch.LongTensor(tg_batch1)

class MyDataSet(Data.Dataset):
    def __init__(self, example):
        super(MyDataSet, self).__init__()
        # self.code_inputs = code_inputs
        # self.desc_inputs = desc_inputs
        # self.target = target
        self.examples = example


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if idx % 2 == 0:
            input_ids = torch.tensor(self.examples[idx].code_ids)
        else:
            input_ids = torch.tensor(self.examples[idx].nl_ids)
        return {
            'input_ids': input_ids, 
            'attention_mask': input_ids.ne(1),
        }

class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize		
        self.indices = range(len(dataset))	 
        self.count = int(len(dataset) / self.batch_size)  

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count
