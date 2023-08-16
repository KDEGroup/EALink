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
newlist = []
dummy_link = pd.read_csv('../data/isis_link.csv',engine='python')
for index, row in dummy_link.iterrows():
    labelist = []
    Diff_processed = []
    difflist = eval(row['Diff'])
    tg = row['label']
    num = len(difflist)
    if tg == 0:
        labelist = [0]*num
    elif tg==1:
        text = str(row['comment']) + str(row['summary']) + str(row['description'])
        text = text.lower()
        cf = eval(row['changed_files'])
        len1 = len(cf)
        if len1 == num:
            for i in range(0,len1):
                func_name = cf[i].split('.')[0].split('/')[-1].lower()
                if text.find(func_name) != -1:
                    labelist.append(1)
                else:
                    labelist.append(0)
        else:
            labelist = [1]*num
    for d in difflist:
        diff = re.sub(r'\<ROW.[0-9]*\>', "",str(d))
        diff = re.sub(r'\<CODE.[0-9]*\>', "",diff)
        diff = re.sub(r'@.*[0-9].*@', "",diff)
        try:
            dl = preprocessor.extract_codetoken(diff, parser,lang)
        except:
            print(dl)
        if len(dl)==0:
            dl =  preprocessor.processDiffCode(diff)
        Diff_processed.append(dl)
    list1 = [Diff_processed,labelist,num]
    newlist.append(list1)
    print(index)
pd.DataFrame(newlist,columns=['Diff_processed','labelist','num']).to_csv("../data/balancedata/java/isis_sub.csv")
