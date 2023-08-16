import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from tqdm import tqdm
import numpy as np
import random
from transformers import (RobertaConfig, RobertaModel, 
                        AdamW, BertConfig, BertModel, BertTokenizer, 
                        RobertaTokenizer,get_linear_schedule_with_warmup, BertForMaskedLM,
                        get_cosine_with_hard_restarts_schedule_with_warmup)
from transformers.modeling_utils import PreTrainedModel
from pathlib import Path
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from dataset import make_data, MyDataSet, MySampler
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
import logging
from transformers import AutoTokenizer, AutoModel
import pandas as pd
logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def set_seed(seed=45):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()


MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

def get_basic_model(config=None):
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    if not config: 
        config = config_class.from_pretrained('microsoft/codebert-base')
    else: config = config 
    config.output_hidden_states = True
    # print(config)
    # tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base')
    nl_model = model_class.from_pretrained('microsoft/codebert-base',
                                    config=config, from_tf=False)
    tokenizer = tokenizer_class.from_pretrained('microsoft/codebert-base', do_lower_case=True)
    nl_model.to(device)
    return nl_model, config, tokenizer

teacher_model, config, tokenizer = get_basic_model()
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
config_path = Path('/tiny_bert_config.json')
student_config = RobertaConfig.from_json_file(config_path) 
student_model, config, tokenizer = get_basic_model(config=student_config)

print('loading data.................')
files = os.listdir('../data/')
corpus_df = pd.DataFrame()
def mergedf(corpus_df, path, files):
    c = 0
    for file in files:
        if file[-8:] == 'link.csv' :
            data = pd.read_csv(path+file)
            corpus_df = pd.concat([corpus_df, data])
    return corpus_df
df = mergedf(corpus_df, '../data/', files)
examples = make_data(df,tokenizer)
train_data = MyDataSet(examples)
my_sampler = MySampler(train_data, 128)
mydataloader = DataLoader(train_data, batch_sampler=my_sampler)
# for step, batch in tqdm(enumerate(mydataloader)):
#     print(batch)
#     exit()

num_epochs = 5
num_training_steps = len(mydataloader) / 16 *num_epochs
print(f'num of training steps:{num_training_steps}')
# Optimizer and learning rate scheduler
optimizer = AdamW(student_model.parameters(), lr=1e-4)

scheduler_class = get_linear_schedule_with_warmup
# arguments dict except 'optimizer'
scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


def simple_adaptor(batch, model_outputs):
    return {'hidden': model_outputs.hidden_states}

distill_config = DistillationConfig(
    intermediate_matches=[    
     {'layer_T': 1, 'layer_S': 0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
     {'layer_T': 4, 'layer_S': 1, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
    #  {'layer_T': 8, 'layer_S': 2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
    #  {'layer_T': 11, 'layer_S': 3, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
    ])
train_config = TrainingConfig()

distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model, 
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

with distiller:
    distiller.train(optimizer, mydataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)
print('codebert')
