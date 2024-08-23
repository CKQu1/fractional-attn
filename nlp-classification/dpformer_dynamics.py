import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from time import time, sleep
from typing import Union
from tqdm import tqdm
#from constants import DROOT, MODEL_NAMES
from constants import *
from mutils import njoin, str2bool, str2ls
from mutils import collect_model_dirs

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

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase
from torch.nn import functional as F  

# True attn score
def get_markov_matrix(C, alpha, bandwidth, d, a):

    #sphere_radius = ((np.pi**(1/d)-1)/np.pi)
    if alpha >= 2:
        alpha_hat = alpha/(alpha-1)
        K = np.exp(-(C/bandwidth**0.5)**alpha_hat)
    else:
        K = (1 + C/bandwidth**0.5)**(-d-alpha)

    D = K.sum(-1)  # row normalization
    K_tilde = np.diag(D**(-a)) @ K @ np.diag(D**(-a))
    D_tilde = K_tilde.sum(-1)

    return K, D, K_tilde, D_tilde

# quick run (single unit)
"""
python -i fns_dynamics.py --models_root=.droot/trained_models_v2/config_qqv/ds\=imdb-layers\=2-heads\=1-hidden\=768-epochs\=5-prj\=qqv/
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='main_seq_classification.py training arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--models_root', default='', help='Pretrained models root')
    parser.add_argument('--N_batch', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, type=bool)

    args = parser.parse_args()    

    if not args.wandb_log:
        os.environ["WANDB_DISABLED"] = "true"

    repo_dir = os.getcwd()  # main dir 
    dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                       if torch.cuda.is_available() else "cpu")   
    # --------------------
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
    # --------------------

    models_root = args.models_root.replace('\\','')
    N_batch = args.N_batch

    # Get model setting from dir
    DCT_ALL = collect_model_dirs(args.models_root, suffix='dp')    
    model_types = list(DCT_ALL.keys())
    assert len(model_types) > 0

    model_info = DCT_ALL[model_types[0]]
    model_dirs = []
    for instance in model_info.loc[0,'instances']:
        model_dirs.append(njoin(model_info.loc[0,'model_dir'], f'model={instance}'))
    
    num_hidden_layers = model_info.loc[0,'num_hidden_layers']

    # Set up plots for later    
    nrows, ncols = 2, 2
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=False,sharey=False)
                            #sharex=True,sharey=True)          
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1) 


    for model_idx, model_dir in enumerate(model_dirs[:1]):        

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

        ##### 0. Model settings #####

        # conditions for diffusion maps
        # assert config['qk_share'], 'QK weight-tying is required!'
        # assert config['num_attention_heads'] == 1, 'Number of attention heads must be 1!'

        # ---------------------------------------- 1. Dataset setup ----------------------------------------
        if model_idx == 0:
            print('---------- Dataset setup ---------- \n')
                    
            if 'fix_embed' in config.keys():
                fix_embed = config['fix_embed']
            else:
                fix_embed = False

            dataset_name = attn_setup['dataset_name']

            def preprocess_function(examples):
                #return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
                return tokenizer(examples['text'], padding=False, truncation=True, max_length=max_length)

            def preprocess_logits_for_metrics(logits, labels):
                preds = logits.argmax(dim=-1)
                return preds    
            
            ###################
            if not fix_embed:
                max_length = config['max_position_embeddings']
                tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                            vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                            merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                            max_length     = max_length)
            else:
                # Load pretrained BERT model and tokenizer
                from transformers import BertModel, BertTokenizer
                
                pretrained_model_name = 'bert-base-uncased'
                tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
                #pretrained_model = BertModel.from_pretrained(pretrained_model_name)  

                max_length = tokenizer.model_max_length - 1

            if dataset_name in ['imdb', 'emotion', 'rotten_tomatoes']:
                def preprocess_function(examples):
                    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
            else:
                def preprocess_function(examples):
                    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=max_length)  

            ###################

            special_tokens = [word for word in tokenizer.get_vocab().keys() if '<' in word and '>' in word]

            # save tokenized dataset
            if not fix_embed:
                tokenized_dataset_dir = njoin(DROOT, "DATASETS", f"tokenized_{dataset_name}-tk=roberta-len={max_length}")
            else:
                tokenized_dataset_dir = njoin(DROOT, "DATASETS", f"tokenized_{dataset_name}-tk={pretrained_model_name}-len={max_length}")

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

        # ---------------------------------------- 2. Load pretrained model ----------------------------------------
        print(f'---------- Load pretrained model ---------- \n')

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
        model.eval() 
        
        c_clears = [0.5, 1]
        markers = ['o', 'x']
        ###### 1. Compute geodesic distances and attention-score ######

        dist_mses = []
        seq_lens = train_dataset['attention_mask'].sum(-1)
        max_seq_len = seq_lens.max()
        idxs_max = torch.argsort(seq_lens, descending=True)
        bidxs = list(idxs_max[:N_batch].numpy())    

        pairwise_overlaps = np.empty([2, num_hidden_layers, N_batch, max_seq_len])  # for attn score and weights     
        pairwise_overlaps[:] = np.nan

        print(f'---------- Begin computations ---------- \n')
        for b_ii, bidx in tqdm(enumerate(bidxs)):         

            ###### 2. Extract single data-point only ######
            X = train_dataset['input_ids'][bidx * train_bs : (bidx+1) * train_bs]  # batch_size must be one if mask is set to None
            Y = train_dataset['label'][bidx * train_bs : (bidx+1) * train_bs]
            bool_mask = train_dataset['attention_mask'][bidx * train_bs : (bidx+1) * train_bs]        
            if train_bs == 1:  # remove padding, only keep the seq from sos to eos        
                X_len = bool_mask.sum().item()
                X = X[:,:X_len]  
                attention_mask = None

            for hidx in range(num_hidden_layers):

                if hidx == 0:
                    pre_sa_hidden_states = model.transformer.embeddings(X)
                else:            
                    pre_sa_hidden_states = model.transformer.encoder.layer[hidx-1](pre_sa_hidden_states)

                if isinstance(pre_sa_hidden_states, tuple):
                    pre_sa_hidden_states = pre_sa_hidden_states[0]    

                hidden_states = pre_sa_hidden_states

                query_vectors = model.transformer.encoder.layer[hidx].attention.self.query(hidden_states)  # (N,B,HD)
                value_vectors = model.transformer.encoder.layer[hidx].attention.self.value(hidden_states)

                batch_size, seq_len, embed_dim = hidden_states.size()
                num_heads, head_dim = model.transformer.encoder.layer[hidx].attention.self.num_heads, model.transformer.encoder.layer[hidx].attention.self.head_dim                
                assert (
                    embed_dim == model.transformer.encoder.layer[hidx].attention.self.embed_dim
                ), f"hidden_states should have embed_dim = {model.transformer.encoder.layer[hidx].attention.self.embed_dim}, but has {embed_dim}"        

                # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)                
                query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)       
                value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                                         


                if not config['qk_share']:
                    query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)
                    key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)     # (B,H,N,D)
                    value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)            
                    att = (query_vectors @ key_vectors.transpose(-2, -1)) * (1.0 / math.sqrt(key_vectors.size(-1)))  # (B,H,N,N)
                    
                else:
                    query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)
                    value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)            
                    att = (query_vectors @ query_vectors.transpose(-2, -1)) * (1.0 / math.sqrt(query_vectors.size(-1)))  # (B,H,N,N)


                if attention_mask is not None:
                    # type 1: key_pad_mask
                    bool_mask = (attention_mask>=0).long()       
                    attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,model.transformer.encoder.layer[hidx].attention.self.num_heads,1,-1])
                    # type 2: symmetrical mask
                    # bool_mask = (attention_mask>=0).long()
                    # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)     

                    # att = att.masked_fill(attention_mask_expanded==0, -1e9)


                # ----- on attn score -----      
                for token_step in range(1, att[0,0].shape[0] - 1):
                    selected_entries = torch.diagonal_scatter(torch.zeros(att[0,0].shape), torch.ones(att[0,0].shape[0] - token_step), token_step)
                    selected_entries += torch.diagonal_scatter(torch.zeros(att[0,0].shape), torch.ones(att[0,0].shape[0] - token_step), -token_step)
                    pairwise_overlaps[0, hidx, b_ii, token_step] = (att[0,0] * selected_entries).mean()


                att = F.softmax(att, dim=-1)
                #att = F.dropout(att, p=self.dropout, training=self.training)  # attn dropout
                attn_output = att @ value_vectors # (B, H, N, N) x (B, H, N, D) -> (B, H, N, D)        
                #attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)  # output dropout
                #assert attn_output.size() == (batch_size, num_heads, seq_len, head_dim), "Unexpected size"
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
                outputs = (attn_output,)

                #return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs                        

                # ----- on attn weights -----      
                for token_step in range(0, att[0,0].shape[0]):
                    selected_entries = torch.diagonal_scatter(torch.zeros(att[0,0].shape), torch.ones(att[0,0].shape[0] - token_step), token_step)
                    if token_step != 0:
                        selected_entries += torch.diagonal_scatter(torch.zeros(att[0,0].shape), torch.ones(att[0,0].shape[0] - token_step), -token_step)
                    pairwise_overlaps[1, hidx, b_ii, token_step] = (att[0,0] * selected_entries).mean()                    

        for hidx in range(num_hidden_layers):
            for ii in range(pairwise_overlaps.shape[0]):
                axs[model_idx, ii].plot(np.arange(max_seq_len), pairwise_overlaps[ii,hidx,].mean(0), label = f'Layer {hidx+1}')
                axs[model_idx, ii].set_xscale('log')
                axs[model_idx, ii].set_yscale('log')

    axs[0,0].legend()

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
    # layers, heads, hidden = config['num_hidden_layers'], config['num_attention_heads'], int(config['hidden_size'])
    # fig_file = 'fdm-'
    # fig_file += f'{model_name}-layers={layers}-heads={heads}-hidden={hidden}-l={hidx}'
    fig_file = 'dp-decay'
    fig_file += '.pdf'
    fig.savefig(njoin(SAVE_DIR, fig_file))            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')    
