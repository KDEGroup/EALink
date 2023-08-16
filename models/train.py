from __future__ import print_function
from __future__ import absolute_import
import argparse
import os
import numpy as np
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, Sampler, random_split
import matplotlib.pyplot as plt
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import logging
from torch.nn import CrossEntropyLoss
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel
from transformers import (RobertaConfig, RobertaModel, 
                        AdamW, BertConfig, BertModel, BertTokenizer, 
                        RobertaTokenizer,get_linear_schedule_with_warmup, BertForMaskedLM,
                        get_cosine_with_hard_restarts_schedule_with_warmup)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
config_path = Path('../Dstill/tiny_bert_config.json')
student_config = RobertaConfig.from_json_file(config_path) 
codeBert = RobertaModel(student_config)
codeBert.load_state_dict(torch.load('../Dstill/saved_models/gs6785.pkl'))



def get_args():
    parser = argparse.ArgumentParser(description="EALink.py")
    parser.add_argument("--end_epoch", type=int, default=400,
                        help="Epoch to stop training.")

    parser.add_argument("--tra_batch_size", type=int, default=16,
                        help="Batch size set during training")

    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="Batch size set during predicting")

    parser.add_argument("--tes_batch_size", type=int, default=16,
                        help="Batch size set during predicting")
    parser.add_argument("--output_model", type=str, default='',
                        help="The path to save model")
    return parser.parse_args()

opt = get_args() 


def text2vec(seqs):
    texttoken_id = []
    max_seq_len = 35
    # print(seqs)
    textoken = []
    for seq in seqs:
        textoken = textoken+[tokenizer.cls_token] + seq
    tokens_ids = tokenizer.convert_tokens_to_ids(textoken)
    texttoken_id = tokens_ids[:max_seq_len]
    texttoken_id.extend([0 for _ in range(max_seq_len -len(texttoken_id))])
    return texttoken_id

def code2vec(codes,isdiff):
    codetoken_id = []
    max_code_len=300
    max_diff_len=500
    codes = eval(codes)
    code_tokens= []
    if isdiff:
        for code in codes:
            code_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + code)
            code_id = code_id[:80]
            code_id.extend([0 for _ in range(80 -len(code_id))])
            codetoken_id.extend(code_id)
        codetoken_id = codetoken_id[:max_diff_len]
        codetoken_id.extend([0 for _ in range(max_diff_len -len(codetoken_id))])
    else:
        for code in codes:
            code_tokens = code_tokens + [tokenizer.cls_token] + code[:80]
        tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)   
        codetoken_id = tokens_ids[:max_code_len]
        codetoken_id.extend([0 for _ in range(max_code_len -len(codetoken_id))])
    # print(len(codetoken_id))
    return codetoken_id


        
def make_batches(df,mode='train'):
    # input_batch, output_batch, target_batch = [],[],[]
    msg_batch1, code_batch1, desc_batch1,tg_batch1, diff_batch1, la_batch1,num_batch1,  commitid_batch1, issueid_batch1= [], [], [], [], [], [], [], [],[]
    logging.info("Loaded the file")
    max_len = 20
    for index, row in df.iterrows():
        commit = text2vec(eval(row['message_processed']))
        issue = text2vec(eval(row['summary_processed'])+eval(row['description_processed']))
        if len(eval(row['codelist_processed']))==0:
            code = code2vec(row['Diff_processed'],False)
        else:
            code = code2vec(row['codelist_processed'],False)
        tg = float(row['target'])
        
        issueid = int(row['issue_id'])
        commitid = row['hash']
        msg_batch1.append(commit)
        code_batch1.append(code)
        desc_batch1.append(issue)
        tg_batch1.append(tg)

        issueid_batch1.append(issueid)
        commitid_batch1.append(commitid)
        diff = code2vec(row['Diff_processed'],True)
        label = eval(row['labelist'])
        label  = label[:max_len]
        label.extend([3 for _ in range(max_len -len(label))])
        num = int(row['num'])
        diff_batch1.append(diff)
        la_batch1.append(label)
        num_batch1.append(num)

    print(len(code_batch1[0]))
    print(len(tg_batch1))

    return torch.LongTensor(msg_batch1), torch.LongTensor(code_batch1), torch.LongTensor(desc_batch1),torch.LongTensor(tg_batch1),torch.LongTensor(diff_batch1),torch.LongTensor(la_batch1),torch.LongTensor(num_batch1),torch.LongTensor(issueid_batch1),commitid_batch1


class AvgPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.code_pooler = AvgPooler()
        self.text_pooler = AvgPooler()

        self.dense = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(768, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class Multi_Model(nn.Module):
    def __init__(self):
        super(Multi_Model, self).__init__()
        self.encoder = codeBert
        self.loss_fct = CrossEntropyLoss()
        self.infonce = InfoNce()
        self.cls = RelationClassifyHeader()

    def getLoss(self, score, t):
        rs = t - score
        rs = torch.abs(rs)
        return torch.sum(rs)

    def get_sim_score(self, text_hidden, code_hidden):
        logits = self.cls(text_hidden=text_hidden, code_hidden=code_hidden)
        sim_scores = torch.softmax(logits, 1).data.tolist()
        return sim_scores
    def forward(self,  message_inputs, code_inputs, desc_inputs, target,querys_inputs=None,diff_inputs=None,label=None,p=None,mode='train'):    # 变动
        t = 0.07
        #data embedding
        commit_inputs = torch.cat([message_inputs, code_inputs],dim=1)
        message = self.encoder(commit_inputs , attention_mask=commit_inputs .ne(1))[0]
        desc = self.encoder(desc_inputs, attention_mask=desc_inputs.ne(1))[0]

        #task1
        commit = self.cls.code_pooler(message)
        issue = self.cls.text_pooler(desc)
        sim = F.cosine_similarity(commit, issue)
        rel_loss = self.getLoss(sim,target)
        #contrastive learning
        commit_vec = F.normalize(commit, p=2, dim=-1, eps=1e-5)
        issue_vec = F.normalize(issue, p=2, dim=-1, eps=1e-5)
        link_repr = torch.cat((issue_vec,commit_vec),1)#[batchsize,2*d_model]
        sims_matrix = torch.matmul(link_repr, link_repr.t())#[batchsize,batchsize]
        sims_matrix = sims_matrix[target == 1]
        cl_loss =  self.infonce(sims_matrix,p,t)
        #task2
        querys = self.encoder(querys_inputs, attention_mask=querys_inputs.ne(1))[0]
        codes = self.encoder(diff_inputs, attention_mask=diff_inputs.ne(1))[0]
        logits = self.cls(code_hidden=codes, text_hidden= querys)
        sub_loss = self.loss_fct(logits.view(-1, 2), label.view(-1))
        loss = rel_loss+cl_loss+sub_loss
        return loss, message,desc   

class MyDataSet(Data.Dataset):
    def __init__(self, message_inputs, code_inputs, desc_inputs,target,diff,label,num,issueid,commitid):
        super(MyDataSet, self).__init__()
        self.message_inputs = message_inputs
        self.code_inputs = code_inputs
        self.desc_inputs = desc_inputs
        self.target = target
        self.diff = diff
        self.label = label
        self.num = num
        self.issueid = issueid
        self.commitid = commitid

    def __len__(self):
        return self.message_inputs.shape[0]

    def __getitem__(self, idx):

        return self.message_inputs[idx], self.code_inputs[idx], self.desc_inputs[idx],self.target[idx],self.diff[idx],self.label[idx],self.num[idx],self.issueid[idx],self.commitid[idx]
class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize		# batchsize
        self.indices = range(len(dataset))	 
        self.count = int(len(dataset) / self.batch_size)  # number of batch

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count

#infonceloss
class InfoNce(nn.Module):
    def __init__(self):
        super(InfoNce, self).__init__()
    
    def forward(self,sims,labels,nce):

        f = lambda x: torch.exp(x/nce)
        new_sim = f(sims)
        return -torch.log(new_sim[labels==1] / new_sim.sum(1)).mean()

def lsplit(list1,n):
    output= [list1[i:i + 80] for i in range(0, 300, 80)]
    return output[:n]

def lremove(list_i,value):
    j = 0 
    for i in range(len(list_i)):
        if list_i[j] == value:
            list_i.pop(j)
        else:
            j += 1
    return list_i

def allindex(data,n):
    index = [i for i, x in enumerate(data) if x == n]
    return index
def allindex_cl(data,n):
    list1 = [0]*16 
    index = [i for i, x in enumerate(data) if x == n]
    kk = random.choice(index)
    list1[kk] = 1
    return list1

if __name__ == '__main__':

    trans_model = Multi_Model()
    trans_model = torch.nn.DataParallel(trans_model).cuda() 
    # adam 
    trans_optimizer = optim.Adam(trans_model.parameters(), lr=4e-05, eps=1e-08)


    df = pd.read_csv('../data/java/calcite_link.csv')
    train_df_sum = df.loc[df['train_flag'] == 1]
    cnt = len(train_df_sum)/4
    train_df = train_df_sum.loc[:cnt*3-1]
    valid_df = train_df_sum.loc[cnt*3:]
    message_input1, code_input1, desc_input1, target1,diff1, label1, nums1,issueid1,commitid1 = make_batches(df=train_df)
    message_input2, code_input2, desc_input2, target2,diff2, label2, nums2,issueid2,commitid2 = make_batches(df=valid_df)

    logging.info("Loaded the file done")
    # data processing
    train_data = MyDataSet(message_input1, code_input1, desc_input1, target1,diff1, label1, nums1, issueid1,commitid1)
    valid_data = MyDataSet(message_input2, code_input2, desc_input2, target2, diff2, label2, nums2,issueid2,commitid2)

    my_sampler1 = MySampler(train_data, opt.tra_batch_size)
    my_sampler2 = MySampler(valid_data, opt.val_batch_size)
    train_data_loader = Data.DataLoader(train_data, batch_sampler=my_sampler1)
    valid_data_loader = Data.DataLoader(valid_data, batch_sampler=my_sampler2)
    best_test_loss = float("inf")

    train_ls, valid_ls = [], []

    for epoch in range(opt.end_epoch):
        epoch_loss = 0
        t3= time.time()
        trans_model.train()
        match_tra_order = 0
        for message_inputs, code_inputs, desc_inputs, target, diff_inputs, labels ,nums,issueid,commitid in train_data_loader:
            difflist = diff_inputs.tolist()
            numlist = nums.tolist()
            desclist = desc_inputs.tolist()
            labelist = labels.tolist()
            issuetest = []
            diff = []
            label = []
            
            idlist = issueid.tolist()
            p = []
            for i in range(0,len(difflist)):
                diff_split = lsplit(difflist[i], numlist[i])
                diff.extend(diff_split)
                len1 = len(diff_split)
                label.extend(lremove(labelist[i],3)[:len1])
                for numx in range(0,len1):
                    issuetest.append(desclist[i])
            idlist = issueid.tolist()
            for j in range(0,len(idlist)):
                if target[j]:
                    p.append(allindex_cl(issueid,idlist[j]))
            issuetest = torch.LongTensor(issuetest)
            diff = torch.LongTensor(diff)
            label = torch.LongTensor(label)
            p = torch.LongTensor(p)
            message_inputs, code_inputs, desc_inputs,target ,issuetest,diff, label, p = message_inputs.cuda(), code_inputs.cuda(), desc_inputs.cuda(), target.cuda(),issuetest.cuda(),diff.cuda(), label.cuda(), p.cuda()
            loss,c_l, n_l = trans_model(message_inputs, code_inputs, desc_inputs,target.float(),issuetest, diff, label, p) 
            trans_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=trans_model.parameters(), max_norm=10, norm_type=2)#要不要删除
            trans_optimizer.step()
            epoch_loss += loss.item()
        train_avg_loss = epoch_loss / len(train_data_loader)
        print(epoch)
        print('\ttrain loss: ', '{:.4f}'.format(train_avg_loss))
        train_ls.append(train_avg_loss)
        torch.cuda.synchronize()
        t4 = time.time()
        print("At the train epoch, cost time:%f" % (t4-t3))


        #eval
        epoch_loss = 0
        trans_model.eval()

        match_val_order = 0
        with torch.no_grad():
            for message_inputs, code_inputs, desc_inputs, target, diff_inputs, labels ,nums,issueid,commitid in valid_data_loader:
                difflist = diff_inputs.tolist()
                numlist = nums.tolist()
                desclist = desc_inputs.tolist()
                labelist = labels.tolist()
                issuetest = []
                diff = []
                label = []
                
                idlist = issueid.tolist()
                p = []
                for i in range(0,len(difflist)):
                    diff_split = lsplit(difflist[i], numlist[i])
                    diff.extend(diff_split)
                    len1 = len(diff_split)
                    label.extend(lremove(labelist[i],3)[:len1])
                    for numx in range(0,len1):
                        issuetest.append(desclist[i])
                idlist = issueid.tolist()
                for j in range(0,len(idlist)):
                    if target[j]:
                        p.append(allindex_cl(issueid,idlist[j]))
                issuetest = torch.LongTensor(issuetest)
                diff = torch.LongTensor(diff)
                label = torch.LongTensor(label)
                p = torch.LongTensor(p)
                message_inputs, code_inputs, desc_inputs,target ,issuetest,diff, label, p = message_inputs.cuda(), code_inputs.cuda(), desc_inputs.cuda(), target.cuda(),issuetest.cuda(),diff.cuda(), label.cuda(), p.cuda()
                loss,c_l, n_l = trans_model(message_inputs, code_inputs, desc_inputs,target.long(),issuetest, diff, label, p) 
                epoch_loss += loss.item()

        valid_avg_loss = epoch_loss / len(valid_data_loader)
        perplexity = math.exp(valid_avg_loss)
        perplexity = torch.tensor(perplexity).item()
        print('\t eval_loss: ', '{:.4f}'.format(valid_avg_loss))
        valid_ls.append(valid_avg_loss)
        print('\tperplexity: ', '{:.4f}'.format(perplexity))         
        torch.save(trans_model.state_dict(), opt.output_model+'model.pt')
