#分开运行 处理分词
import pandas as pd
import numpy as np
from numpy import nan
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import preprocessor
import re
import csv 
from tqdm import tqdm
from tree_sitter import Language, Parser
from queue import Queue
from keras.preprocessing.text import Tokenizer
from parser_lang import (tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
import re
import pickle
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
import gc
from tkinter import _flatten
csv.field_size_limit(500 * 1024 * 1024)
lang = 'java'

LANGUAGE = Language('/parser_lang/my-languages.so', lang)
parser = Parser()
parser.set_language(LANGUAGE)
process = []
dummy_link = pd.read_csv('../data/train_flag/isis_link.csv',engine='python')
# data1 = dummy_link.get_chunk(1000)
# print(data1.columns)
dummy_link.rename(columns={'commitid':'hash'},inplace=True)
for index, row in dummy_link.iterrows():
    summary_processed = preprocessor.preprocessNoCamel(str(row['summary']).strip('[]'))
    description_processed = preprocessor.preprocessNoCamel(str(row['description']).strip('[]'))
    message_processed = preprocessor.preprocessNoCamel(str(row['message']).strip('[]'))
    diff = re.sub(r'\<ROW.[0-9]*\>', "",str(row['Diff']))
    diff = re.sub(r'\<CODE.[0-9]*\>', "",diff)
    Diff_processed = preprocessor.processDiffCode(diff)
    changed_files = []
    cf = eval(row['changed_files'])
    for f in cf:
        f_name = f.split('/')[-1]
        changed_files.append(f_name)
    clist = eval(row['codelist'])
    codelist_processed = []
    for code in clist:
        codelist_processed.append(preprocessor.extract_codetoken(code, parser,lang))
    issue_text = str(row['summary'])+str(row['description'])+str(row['comment'])#【待加入comment】
    issuecode = preprocessor.getIssueCode(issue_text)
    list1 = [row['source'],row['product'],row['issue_id'],row['component'],row['creator_key'],row['create_date'],row['update_date'],row['last_resolved_date'],summary_processed,description_processed,issuecode,row['issue_type'],row['status'],row['repo'],row['hash'],row['parents'],row['author'],row['committer'],row['author_time_date'],row['commit_time_date'],message_processed,row['commit_issue_id'],changed_files,Diff_processed,codelist_processed,row['label'],row['train_flag']]
    process.append(list1)
    print(index)
pd.DataFrame(process,columns=['source', 'product', 'issue_id', 'component', 'creator_key',
       'create_date', 'update_date', 'last_resolved_date', 'summary_processed',
       'description_processed','issuecode','issue_type', 'status', 'repo', 'hash',
       'parents', 'author', 'committer', 'author_time_date',
       'commit_time_date', 'message_processed', 'commit_issue_id',
       'changed_files', 'Diff_processed', 'codelist_processed', 'label',
       'train_flag']).to_csv("../data/balancedata/java/isis_link_process.csv")
