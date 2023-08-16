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
    parser = argparse.ArgumentParser(description="test_EALink.py")

    parser.add_argument("--tes_batch_size", type=int, default=16,
                        help="Batch size set during predicting")
    parser.add_argument("--model_path", type=str, default='',
                        help="the path of EALink model")
    return parser.parse_args()

opt = get_args() 

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


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


        
def make_batches(df):
    # input_batch, output_batch, target_batch = [],[],[]
    msg_batch1, code_batch1, desc_batch1,tg_batch1, diff_batch1, la_batch1,num_batch1,  commitid_batch1, issueid_batch1= [], [], [], [], [], [], [], [],[]
    logging.info("Loaded the file")
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
    print(len(code_batch1[0]))
    print(len(tg_batch1))
    return torch.LongTensor(msg_batch1), torch.LongTensor(code_batch1), torch.LongTensor(desc_batch1),torch.LongTensor(tg_batch1),torch.LongTensor(issueid_batch1),commitid_batch1


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
    def forward(self,  message_inputs, code_inputs, desc_inputs, target,querys_inputs=None,diff_inputs=None,label=None,p=None,mode='train'):   
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
        if mode=='test':
            return rel_loss, sim.data.tolist()
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

class TestDataSet(Data.Dataset):
    def __init__(self, message_inputs, code_inputs, desc_inputs,target,issueid,commitid):
        super(TestDataSet, self).__init__()
        self.message_inputs = message_inputs
        self.code_inputs = code_inputs
        self.desc_inputs = desc_inputs
        self.target = target
        self.issueid = issueid
        self.commitid = commitid

    def __len__(self):
        return self.message_inputs.shape[0]

    def __getitem__(self, idx):

        return self.message_inputs[idx], self.code_inputs[idx], self.desc_inputs[idx],self.target[idx],self.issueid[idx],self.commitid[idx]

class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize		# batchsize
        self.indices = range(len(dataset))	 
        self.count = int(len(dataset) / self.batch_size)  # number of batchsize

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
def get_retrieval(df):
    test_true_link = df.loc[df['target']==1]
    false_link = df.loc[df['target']==0]
    test_df = test_true_link[['issue_id', 'summary_processed','description_processed','issuecode','hash','message_processed', 'changed_files', 'Diff_processed', 'codelist_processed','target']]
    issue_id_list = test_true_link['issue_id'].unique()
    cols = ['issue_id', 'summary_processed','description_processed','issuecode']
    cols2 = ['hash','message_processed', 'changed_files', 'Diff_processed', 'Diff_processed']
    for idx in issue_id_list[:1000]:
        temp = test_true_link.loc[test_true_link['issue_id']==idx].reset_index(drop=True)
        issue = pd.DataFrame(temp[cols].loc[0,:])
        issue = pd.DataFrame(issue.values.T, index=issue.columns, columns=issue.index)
        falsel = false_link.loc[df['issue_id']==idx]
        falsel = falsel[['issue_id', 'summary_processed','description_processed','issuecode','hash','message_processed', 'changed_files', 'Diff_processed', 'codelist_processed','target']]
        len1 = len(falsel)
        if len1:
            test_df = test_df.append(falsel, ignore_index=True)
        commit = test_true_link.loc[test_true_link['issue_id']!=idx].reset_index(drop=True).loc[:99-len1,cols2]
        for j in range(len(commit)):
            y = pd.DataFrame(commit.iloc[j])
            y = pd.DataFrame(y.values.T, index=y.columns, columns=y.index)
            y = y.reset_index(drop=True)
            x = issue
            resulted_false_link = x.join(y)
            resulted_false_link.insert(loc=len(resulted_false_link.columns),column='target',value=0)
            resulted_false_link = resulted_false_link.loc[ : , ~resulted_false_link.columns.str.contains('Unnamed')]
            test_df = test_df.append(resulted_false_link, ignore_index=True)
        print(len1)
        print(len(commit))

    print(len(test_df))
    test_df.to_csv('../data/test/calcite_test.csv')
    return test_df

def NDCG_at_K(data_frame, k=1):
    group_tops = data_frame.groupby('s_id')
    cnt = 0
    dcg_sum = 0
    for s_id, group in group_tops:
        rank = 0
        for index, row in group.head(k).iterrows():
            rank += 1
            if row['label'] == 1:
                dcg_sum += math.log(2)/math.log(rank+2) 
                break 
        cnt += 1
    return round(dcg_sum / cnt if cnt > 0 else 0, 4)

def recall_at_K(data_frame, k=1):
    group_tops = data_frame.groupby('s_id')
    df = data_frame.loc[data_frame['label']==1]
    cnt = 0
    recall = 0.0
    for s_id, group in group_tops:
        hits = 0
        tu = df.loc[df['s_id']==s_id]
        for index, row in group.head(k).iterrows():
            hits += 1 if row['label'] == 1 else 0      
        recall += round(hits / len(tu) if len(tu) > 0 else 0, 4)
        cnt +=1
    return recall/cnt

def precision_at_K(data_frame, k=1):
    group_tops = data_frame.groupby('s_id')
    cnt = 0
    hits = 0
    for s_id, group in group_tops:
        for index, row in group.head(k).iterrows():
            hits += 1 if row['label'] == 1 else 0      
        cnt += k
    return round(hits / cnt if cnt > 0 else 0, 4)
def Hit_at_K(data_frame, k=1):
    group_tops = data_frame.groupby('s_id')
    cnt = 0
    hits = 0
    for s_id, group in group_tops:
        for index, row in group.head(k).iterrows():
            if row['label'] == 1:
                hits += 1 
                break       
        cnt += 1
    return round(hits / cnt if cnt > 0 else 0, 4)

def MRR(data_frame):
    group_tops = data_frame.groupby('s_id')
    mrr_sum = 0
    for s_id, group in group_tops:
        rank = 0
        for i, (index, row) in enumerate(group.iterrows()):
            rank += 1
            if row['label'] == 1:
                mrr_sum += 1.0 / rank
                break
    return mrr_sum / len(group_tops)
def results_to_df(res: List[Tuple]) -> DataFrame:

    df = pd.DataFrame()
    df['s_id'] = [x[0] for x in res]
    df['t_id'] = [x[1] for x in res]
    df['pred'] = [x[2] for x in res]
    df['label'] = [x[3] for x in res]
    group_sort = df.groupby(["s_id"]).apply(
        lambda x: x.sort_values(["pred"], ascending=False)).reset_index(drop=True)
    return group_sort

if __name__ == '__main__':

    trans_model = Multi_Model()
    trans_model = torch.nn.DataParallel(trans_model).cuda() 
    # adam 
    trans_optimizer = optim.Adam(trans_model.parameters(), lr=4e-05, eps=1e-08)

    df = pd.read_csv('../data/calcite_link.csv')
    test_t = df.loc[df['train_flag'] == 0]
    test_df = get_retrieval(test_t)
    message_input3, code_input3, desc_input3, target3,issueid3,commitid3 = make_batches(df=test_df)

    logging.info("Loaded the file done")
    # data processing
    test_data = TestDataSet(message_input3, code_input3, desc_input3, target3, issueid3,commitid3)
    my_sampler3 = MySampler(test_data, opt.tes_batch_size)
    test_data_loader = Data.DataLoader(test_data, batch_sampler=my_sampler3)
    best_test_loss = float("inf")

    # Test
    retr_res_path = '../data/res/res_calcite.csv'
    state_dict = torch.load(opt.model_path)
    trans_model.load_state_dict(state_dict)
    trans_model.eval()
    res = []
    mrr = 0
    hit1 = 0
    hit10 = 0
    cnt = 0
    p10= 0
    with torch.no_grad():
        trans_model.eval()
        for message_inputs, code_inputs, desc_inputs, target, issueid,commitid in test_data_loader:
            # print("inputs shape:",inputs1.shape)
            message_inputs, code_inputs, desc_inputs,target = message_inputs.cuda(), code_inputs.cuda(), desc_inputs.cuda(), target.cuda()
            loss,sim_score = trans_model(message_inputs, code_inputs, desc_inputs,target.long(),mode='test') 
            for n, p, prd, lb in zip(issueid.tolist(), list(commitid), sim_score, target.tolist()):
                res.append((n, p, prd, lb))
    df = results_to_df(res)
    pd.DataFrame(df)
    df.reset_index(inplace=True)
    path = Path(retr_res_path)
    df.to_csv(path)

    Hit = Hit_at_K(df, 1)
    print("  Final test Hit@1 %f" % (Hit) )
    Hit5 = Hit_at_K(df, 5)
    print("  Final test Hit@5 %f" % (Hit5) )
    Hit10 = Hit_at_K(df, 10)
    print("  Final test Hit@10 %f" % (Hit10) )
    Hit20 = Hit_at_K(df, 20)
    print("  Final test Hit@20 %f" % (Hit20) )
    precision = precision_at_K(df, 1)
    print("  Final test precision@1 %f" % (precision) )  
    precision5 = precision_at_K(df, 5)
    print("  Final test precision@5 %f" % (precision5))
    precision10 = precision_at_K(df, 10)
    print("  Final test precision@10 %f" % (precision10))
    precision20 = precision_at_K(df, 20)
    print("  Final test precision@20 %f" % (precision20))
    recall = recall_at_K(df, 1)
    print("  Final test recall@1 %f" % (recall))  
    recall5 = recall_at_K(df, 5)
    print("  Final test recall@5 %f" % (recall5))  
    recall10 = recall_at_K(df, 10)
    print("  Final test recall@10 %f" % (recall10))
    recall20 = recall_at_K(df, 20)
    print("  Final test recall@20 %f" % (recall20))
    mrr = MRR(df)
    print("  Final test MRR %f" % (mrr)) 
    ngcg = NDCG_at_K(df, k=1)
    print("  Final test ndcg@1 %f" % (ngcg))
    ngcg5 = NDCG_at_K(df, k=5)
    print("  Final test ndcg@5 %f" % (ngcg5))
    ngcg10 = NDCG_at_K(df, k=10)
    print("  Final test ndcg@10 %f" % (ngcg10))  
    ngcg20 = NDCG_at_K(df, k=20)
    print("  Final test ndcg@20 %f" % (ngcg20))        
  
