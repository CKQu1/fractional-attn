import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from time import time, sleep
from typing import Union
from tqdm import tqdm
from constants import DROOT, MODEL_NAMES
from mutils import njoin
from data_utils import get_dataset, get_dataset_cols, process_dataset_cols

from os import makedirs
from os.path import isdir, isfile
#from sklearn.metrics import f1_score
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import RobertaTokenizer
from transformers import AdamW
from transformers.utils import logging
from transformers.trainer_pt_utils import get_parameter_names

from datasets import load_dataset, load_metric, load_from_disk
from models.model_app import FNSFormerForSequenceClassification
from models.model_utils import ModelConfig

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings("ignore")    

# quick run (single unit)
"""
python -i fns_dynamics.py --model_dir=.droot/debug-mode/rdfnsformer-rt-alpha=1.5-eps=1-a=1/model=0/
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='main_seq_classification.py training arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--model_dir', default='', type=str, help='Pretrained model path')
    parser.add_argument('--wandb_log', default=False, type=bool)

    args = parser.parse_args()    

    if not args.wandb_log:
        os.environ["WANDB_DISABLED"] = "true"

    repo_dir = os.getcwd()  # main dir 
    dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                       if torch.cuda.is_available() else "cpu")   
    device_name = "GPU" if dev.type != "cpu" else "CPU"
    #ddp = torch.distributed.is_available()
    ddp = False
    global_rank = None
    if ddp:
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        print(f"global_rank: {global_rank}")    
        print(f"Device in use: {dev}.")
        device_total = world_size
        master_process = global_rank == 0 # this process will do logging, checkpointing etc.             
    else:        
        device_total = 1       
        master_process = True    

    logging.set_verbosity_debug()
    logger = logging.get_logger()    

    model_dir = args.model_dir
    train_setting = pd.read_csv(njoin(model_dir, 'train_setting.csv'))
    if os.path.isdir(njoin(model_dir, 'run_performance.csv')):
        run_performance = pd.read_csv(njoin(model_dir, 'run_performance.csv'))
    train_setting = pd.read_csv(njoin(model_dir, 'train_setting.csv'))
    if os.path.isdir(njoin(model_dir, 'run_performance.csv')):
        run_performance = pd.read_csv(njoin(model_dir, 'run_performance.csv'))
    f = open(njoin(model_dir,'config.json'))
    config = json.load(f)
    f.close()
    f = open(njoin(model_dir,'attn_setup.json'))
    attn_setup = json.load(f)
    f.close()   

    # conditions for diffusion maps
    assert config['qk_share'], 'QK weight-tying is required!'
    assert 'op' in attn_setup['model_name'], 'Orthogonal projection matrix is required!'

    # ---------------------------------------- 1. Dataset setup ----------------------------------------
    print('---------- Dataset setup ----------')
   
    max_length = config['attention_window']
    dataset_name = attn_setup['dataset_name']

    def preprocess_function(examples):
        #return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
        return tokenizer(examples['text'], padding=False, truncation=True, max_length=max_length)

    def preprocess_logits_for_metrics(logits, labels):
        preds = logits.argmax(dim=-1)
        return preds    
    
    tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                 vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                 merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                 max_length     = max_length)    

    special_tokens = [word for word in tokenizer.get_vocab().keys() if '<' in word and '>' in word]

    # save tokenized dataset
    tokenized_dataset_dir = njoin(DROOT, "DATASETS", f"tokenized_{dataset_name}")
    if not isdir(tokenized_dataset_dir):         
        print("Downloading data!") if master_process else None
        # create cache for dataset
        dataset = get_dataset(dataset_name, njoin(DROOT, "DATASETS"))

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        column_names = get_dataset_cols(tokenized_dataset)
        if 'text' in column_names:
            tokenized_dataset = tokenized_dataset.map(remove_columns=['text'])
        if not isdir(tokenized_dataset_dir): os.makedirs(tokenized_dataset_dir)
        tokenized_dataset.save_to_disk(tokenized_dataset_dir)
        del dataset  # alleviate memory
    else:        
        print("Data downloaded, loading from local now! \n") if master_process else None
        tokenized_dataset = load_from_disk(tokenized_dataset_dir)
    if attn_setup['dataset_name'] != 'imdb':
        tokenized_dataset = process_dataset_cols(tokenized_dataset)

    keys = list(tokenized_dataset.keys())
    if len(keys) == 1:
        tokenized_dataset = tokenized_dataset[keys[0]].train_test_split(0.5)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]             

    # convert to torch.Tensor from list
    train_dataset.set_format('torch')
    eval_dataset.set_format('torch')

    # data
    dataset = attn_setup['dataset_name']
    #train_bs = int(train_setting.loc[0,'batch_size'])
    train_bs = 1
    eval_bs = 1
    N_batch = 1

    # ---------------------------------------- 2. Load pretrained model ----------------------------------------
    print('---------- Load pretrained model ----------')

    # model
    model_name = attn_setup['model_name']
    assert model_name in MODEL_NAMES, f'{model_name} does not exist in {MODEL_NAMES}'

    model_config = ModelConfig.from_json_file(njoin(model_dir, 'config.json'))
    model = FNSFormerForSequenceClassification(model_config, **attn_setup).to(dev)
    checkpoints = []
    checkpoint_steps = []
    for subdir in os.listdir(model_dir):
        if 'checkpoint-' in subdir:
            checkpoints.append(njoin(model_dir,subdir))
            checkpoint_steps.append(int(subdir.split('-')[-1]))
    checkpoint = torch.load(njoin(model_dir, f'checkpoint-{checkpoint_steps[-1]}', 'pytorch_model.bin'), map_location=dev)
    model.load_state_dict(checkpoint)
    
    import math
    from torch.nn import functional as F    
    
    for batch_idx in range(N_batch):

        X = train_dataset['input_ids'][batch_idx * train_bs : (batch_idx+1) * train_bs]  # batch_size must be one if jask is set to None
        bool_mask = train_dataset['attention_mask'][batch_idx * train_bs : (batch_idx+1) * train_bs]        
        if train_bs == 1:
            # remove padding
            X_len = bool_mask.sum().item()
            X = X[:,:X_len]  

        hidden_states = model.transformer.embeddings(X)

        alpha = attn_setup['alpha']
        bandwidth = attn_setup['bandwidth']
        a = attn_setup['a']

        if alpha < 2:
            d_intrinsic = attn_setup['d_intrinsic']

        # (N,B,HD)
        query_vectors = model.transformer.encoder.layer[0].attention.self.query(hidden_states)
        value_vectors = model.transformer.encoder.layer[0].attention.self.value(hidden_states)

        #seq_len, batch_size, embed_dim = hidden_states.size()
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = model.transformer.encoder.layer[0].attention.self.num_heads, model.transformer.encoder.layer[0].attention.self.head_dim                
        assert (
            embed_dim == model.transformer.encoder.layer[0].attention.self.embed_dim
        ), f"hidden_states should have embed_dim = {model.transformer.encoder.layer[0].attention.self.embed_dim}, but has {embed_dim}"


        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        if alpha < 2:
            query_vectors = query_vectors.view(batch_size, seq_len, num_heads, d_intrinsic).transpose(1, 2)
        else:
            query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)
        #query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        if not config['qk_share']:
            key_vectors = model.transformer.encoder.layer[0].attention.self.key(hidden_states)
            if alpha < 2:
                key_vectors = key_vectors.view(batch_size, seq_len, num_heads, d_intrinsic).transpose(1, 2)
            else:      
                key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B,H,N,D)   

            Dist = torch.cdist(query_vectors, key_vectors, p=2)
        else:
            Dist = torch.cdist(query_vectors, query_vectors, p=2)        

        # type 1: key_pad_mask
        # bool_mask = (attention_mask>=0).long()
        if train_bs == 1:
            attention_mask_expanded = None
        else:            
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,model.transformer.encoder.layer[0].attention.self.num_heads,1,-1])

        # type 2: symmetrical mask
        # bool_mask = (attention_mask>=0).long()
        # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)     

        #g_dist = g_dist.masked_fill(attention_mask_expanded==0, 1e5)        

        if alpha < 2:
            # g_dist = Dist / head_dim  # (H,B,N,N)
            g_dist = Dist * (2**(1/head_dim) - 1) / head_dim
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            g_dist = Dist / head_dim**0.5  # (H,B,N,N)
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))
        #attn_score = attn_score.masked_fill(attention_mask_expanded==0, -1e9)

        attn_score_shape = attn_score.shape
        #bound = 1e9 * seq_len
        bound = 1e5    
        if a > 0:
            
            # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            # N_R = torch.clamp(attn_score.sum(-1), min=1/bound, max=bound)  # row sum
            # N_C = torch.clamp(attn_score.sum(-2), min=1/bound, max=bound)  # col sum
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)

            # added for analysis
            with torch.no_grad():
                N_R_hat = K_tilde.sum(-1)
                N_C_hat = K_tilde.sum(-2)
                K_hat = N_R_hat * K_tilde * N_C_hat
                K_hat = 1/2 * (K_hat + K_hat.transpose(2,3))    
                L, V = torch.linalg.eig(K_hat)        

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive     

        attn_weights = F.dropout(attn_weights, p=model.transformer.encoder.layer[0].attention.self.dropout, 
                                training=model.transformer.encoder.layer[0].attention.self.training)   

        attn_output = attn_weights @ value_vectors  

    # ---------------------------------------- 3. Diagonalization ----------------------------------------


