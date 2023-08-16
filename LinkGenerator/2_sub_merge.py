import os
import ast
import torch as th
import dgl
from dgl.data.utils import save_graphs, load_graphs
import datetime
import astunparse
import random
from gensim.models import word2vec, KeyedVectors, Word2Vec
import pandas as pd
import itertools
import os
from numpy import nan as NaN
import gc
prolist = ['isis']
for pro in prolist:
    df1 = pd.read_csv('../data/balancedata/java/isis_link_process.csv')
    df2 = pd.read_csv('../data/balancedata2/java/isis_sub.csv')
    list1 = ['issue_id','summary_processed','description_processed','issuecode','hash','message_processed','changed_files','codelist_processed','label','train_flag']
    df = df1[list1]
    df.insert(len(df.columns),'Diff_processed',value=NaN)
    df.insert(len(df.columns),'labelist',value=NaN)
    df.insert(len(df.columns),'num',value=NaN)
    tg = df.label
    res = df.drop('label',axis=1)
    res.insert(len(res.columns),'target',tg)  
    flag = res.train_flag
    res = res.drop('train_flag',axis=1)
    res.insert(len(res.columns),'train_flag',flag)  
    res['Diff_processed'] = df2.Diff_processed
    res['labelist'] = df2.labelist
    res['num'] = df2.num
    res = res.loc[ : , ~res.columns.str.contains('Unnamed')]
    res.to_csv('../data/isis_link.csv')
    print(res.columns)




