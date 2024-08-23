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
from mutils import *
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
from matplotlib.cm import get_cmap

from string import ascii_lowercase
from torch.nn import functional as F  

# ----- global plot settings -----
c_clears = [0.5, 1]
markers = ['o', 'x']

cmap_attn = get_cmap('inferno')
#cmap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
# --------------------------------

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
    parser.add_argument('--N_batch', default=1, type=int)
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

    # Get model setting from dir
    models_root = args.models_root.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(models_root) if 'former' in dirname])  

    # select based on bandwidth
    suffix = MODEL_SUFFIX
    DCT_ALL = collect_model_dirs(models_root, suffix=suffix)
    model_types = list(DCT_ALL.keys())

    SELECTED_ALPHAS = [1.2,1.6,2.0]
    for model_type in model_types:
        if 'fns' in model_type:
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            # ----- filter alphas -----
            df_model = df_model[df_model['alpha'].isin(SELECTED_ALPHAS)]
            # ------------------------
            df_model.reset_index(drop=True, inplace=True)
            break

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = df_model.loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]
    dataset = df_model.loc[0,'dataset_name']

    # ----- fns setting -----
    #alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    alphas = sorted(df_model.loc[:,'alpha'].unique())  # small to large
    epss = sorted(df_model.loc[:,'bandwidth'].unique())  

    model_dirs = []
    for ii in range(df_model.shape[0]):
        ensembles = df_model.loc[ii,'ensembles']
        if ensembles > 0:
            instance = str(df_model.loc[ii,'instances'][0])
            model_dirs.append(njoin(df_model.loc[ii,'model_dir'], f'model={instance}'))

            f = open(njoin(model_dirs[0],'config.json'))
            config = json.load(f)
            f.close()
            f = open(njoin(model_dirs[0],'attn_setup.json'))
            attn_setup = json.load(f)
            f.close()  
    model_dirs = sorted(model_dirs)

    num_hidden_layers = config['num_hidden_layers']
    EDs = np.zeros([num_hidden_layers, len(alphas), len(epss)])

    # Set up plots for later    
    #nrows, ncols = 1 + len(alphas), len(epss)
    nrows, ncols = len(alphas), len(epss)
    figsize = (3*ncols,3*nrows)

    figs, axss = [], []
    for _ in range(num_hidden_layers):
        fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                                sharex=False,sharey=False)
                                #sharex=True,sharey=True)          
        if nrows == 1:
            if ncols > 1:
                axs = np.expand_dims(axs, axis=0)
            else:
                axs = np.expand_dims(axs, axis=[0,1])     
        axs = axs.flatten()    

        figs.append(fig); axss.append(axs)

    attn_weightss = []      
    K_tildes = []
    for model_idx, model_dir in enumerate(model_dirs):        

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
        alpha = attn_setup['alpha']
        bandwidth = attn_setup['bandwidth']
        a = attn_setup['a']
        if alpha < 2:
            d_intrinsic = attn_setup['d_intrinsic']   
        mr_ii = epss.index(bandwidth)

        # conditions for diffusion maps
        assert config['qk_share'], 'QK weight-tying is required!'
        assert 'opfns' in attn_setup['model_name'], 'Orthogonal projection matrix is required!'
        assert config['num_attention_heads'] == 1, 'Number of attention heads must be 1!'

        # ---------------------------------------- 1. Dataset setup ----------------------------------------
        if model_idx == 0:
            print('---------- Dataset setup ----------')
                    
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
            #batch_size = int(train_setting.loc[0,'batch_size'])
            batch_size = 1                

            N_batch = args.N_batch
            seq_lens = train_dataset['attention_mask'].sum(-1)
            max_seq_len = seq_lens.max()

            #idxs_max = torch.argsort(seq_lens, descending=True)  # long to short sequences
            idxs_max = torch.argsort(seq_lens, descending=False)  # short to long sequences
            seq_lens = seq_lens[idxs_max]

            min_len = 50
            if min_len is not None:
                bidxs = list(idxs_max[seq_lens>=min_len][:N_batch].numpy())
            else:
                bidxs = list(idxs_max[:N_batch].numpy())

        # ---------------------------------------- 2. Load pretrained model ----------------------------------------
        print(f'---------- Load pretrained model: (alpha, eps) = ({alpha}, {bandwidth}) ----------')

        # model
        model_name = attn_setup['model_name']    
        assert model_name in MODEL_NAMES, f'{model_name} does not exist in {MODEL_NAMES}'

        if 'fns' in model_name:
            manifold = attn_setup['manifold']
            sphere_radius = config['sphere_radius']
            mask_val = config['mask_val']
        else:
            manifold = None

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
     
        ###### 1. Compute geodesic distances and attention-score ######

        pairwise_overlaps = np.empty([2, num_hidden_layers, N_batch, max_seq_len])  # for attn score and weights     
        pairwise_overlaps[:] = np.nan        
        for b_ii, bidx in tqdm(enumerate(bidxs)):         

            ###### 2. Extract single data-point only ######
            X = train_dataset['input_ids'][bidx * batch_size : (bidx+1) * batch_size]  # batch_size must be one if mask is set to None
            Y = train_dataset['label'][bidx * batch_size : (bidx+1) * batch_size]
            bool_mask = train_dataset['attention_mask'][bidx * batch_size : (bidx+1) * batch_size]        
            if batch_size == 1:  # remove padding, only keep the seq from sos to eos        
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

                if manifold == 'sphere':
                    hidden_states = F.normalize(pre_sa_hidden_states,p=2,dim=-1)
                elif manifold == 'rd':
                    hidden_states = pre_sa_hidden_states

                #query_vectors = model.transformer.encoder.layer[hidx].attention.self.query(hidden_states)  # (N,B,HD)
                query_vectors = hidden_states
                value_vectors = model.transformer.encoder.layer[hidx].attention.self.value(hidden_states)

                batch_size, seq_len, embed_dim = hidden_states.size()
                num_heads, head_dim = model.transformer.encoder.layer[hidx].attention.self.num_heads, model.transformer.encoder.layer[hidx].attention.self.head_dim                
                assert (
                    embed_dim == model.transformer.encoder.layer[hidx].attention.self.embed_dim
                ), f"hidden_states should have embed_dim = {model.transformer.encoder.layer[hidx].attention.self.embed_dim}, but has {embed_dim}"        

                # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)                
                query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)       
                value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                                         

                # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
                if not config['qk_share']:
                    key_vectors = model.transformer.encoder.layer[hidx].attention.self.key(hidden_states)    
                    key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,H,N,D)   

                    if manifold == 'rd':
                        Dist = torch.cdist(query_vectors, key_vectors, p=2)
                    elif manifold == 'sphere':
                        eps = 1e-7  # for limiting the divergence from acos                                                
                        Dist = torch.acos(torch.clamp(query_vectors @ key_vectors.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius               
                else:
                    if manifold == 'rd':
                        Dist = torch.cdist(query_vectors, query_vectors, p=2)     
                    elif manifold == 'sphere':
                        eps = 1e-7  # for limiting the divergence from acos                        
                        Dist = torch.acos(torch.clamp(query_vectors @ query_vectors.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius                                                                 

                if attention_mask is not None:
                    # type 1: key_pad_mask
                    bool_mask = (attention_mask>=0).long()       
                    attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,model.transformer.encoder.layer[hidx].attention.self.num_heads,1,-1])
                    # type 2: symmetrical mask
                    # bool_mask = (attention_mask>=0).long()
                    # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)     

                #g_dist = g_dist.masked_fill(attention_mask_expanded==0, 1e5)                   

                if alpha < 2:
                    # g_dist = Dist / head_dim  # (H,B,N,N)
                    if manifold == 'rd':
                        g_dist = Dist * (2**(1/head_dim) - 1) / head_dim    
                    elif manifold == 'sphere':
                        g_dist = Dist
                    attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
                else:
                    if manifold == 'rd':
                        g_dist = Dist / head_dim**0.5  # (H,B,N,N)
                    elif manifold == 'sphere':
                        g_dist = Dist                
                    attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))

                #attn_score = attn_score.masked_fill(attention_mask_expanded==0, -1e9)
                #assert (attn_score==attn_score.transpose(2,3)).sum() == attn_score.numel()

                for token_step in range(1, attn_score[0,0].shape[0] - 1):
                    selected_entries = torch.diagonal_scatter(torch.zeros(attn_score[0,0].shape), torch.ones(attn_score[0,0].shape[0] - token_step), token_step)
                    selected_entries += torch.diagonal_scatter(torch.zeros(attn_score[0,0].shape), torch.ones(attn_score[0,0].shape[0] - token_step), -token_step)
                    pairwise_overlaps[0, hidx, b_ii, token_step] = (attn_score[0,0] * selected_entries).mean()

                ###### 3. Extended analysis on removing non-uniform sampling (MC equiv to attn-score) ######

                # K_tilde = K_tilde[0,0]
                # D_tilde = K_tilde.sum(-1)        
                # # can do this as the attn weights are always positive  
                # MC = torch.diag(D_tilde**(-1)) @ K_tilde  # same as MC_ = F.normalize(K_tilde,p=1,dim=-1)

                # K_hat_ = torch.diag(D_tilde_**(-0.5)) @ K_tilde_ @ torch.diag(D_tilde_**(-0.5))
                # K_hat_sym_ = 0.5*(K_hat_ + K_hat_.T)
                # eigvals_, eigvecs_ = torch.linalg.eigh(K_hat_sym_)                                
                # eigvals_, eigvecs_ = torch.linalg.eigh(MC_)
                # eigvals_ = eigvals_.detach().numpy()
                # eigvecs_ = eigvecs_.detach().numpy()
                # # order based on eigvals from small to large
                # ii = np.argsort(eigvals_[0,0])
                # eigvals_ = eigvals_[:,:,ii]
                # eigvecs_ = eigvecs_[:,:,:,ii]
                
                ###### 4. Obtain attention weights based on model settings ######

                if a > 0:            
                    N_R = attn_score.sum(-1)  # row sum
                    N_C = attn_score.sum(-2)  # col su                
                    K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)       
                else:
                    K_tilde = attn_score

                K_tilde = K_tilde[0,0]
                D_tilde = K_tilde.sum(-1)

                attn_weights = F.normalize(K_tilde,p=1,dim=-1)  # can do this as the attn weights are always positive     
                # attn_weights = F.dropout(attn_weights, p=model.transformer.encoder.layer[hidx].attention.self.dropout, 
                #                          training=model.transformer.encoder.layer[hidx].attention.self.training)           

                # for token_step in range(1, attn_weights.shape[0] - 1):
                #     selected_entries = torch.diagonal_scatter(torch.zeros(attn_weights.shape), torch.ones(attn_weights.shape[0] - token_step), token_step)
                #     selected_entries += torch.diagonal_scatter(torch.zeros(attn_weights.shape), torch.ones(attn_weights.shape[0] - token_step), -token_step)
                #     pairwise_overlaps[1, hidx, b_ii, token_step] = (attn_weights * selected_entries).mean()

                attn_weightss.append(attn_weights)
                K_tildes.append(K_tilde)

                alpidx = alphas.index(alpha)
                row = alpidx + 1

                """
                K_hat = torch.diag(D_tilde**(-0.5)) @ K_tilde @ torch.diag(D_tilde**(-0.5))
                K_hat_sym = 0.5*(K_hat + K_hat.T)
                eigvals, eigvecs = torch.linalg.eigh(K_hat_sym)    
                eigvecs = torch.diag(D_tilde**(-0.5)) @ eigvecs

                eigvals = eigvals.detach().numpy()
                eigvecs = eigvecs.detach().numpy()      
                # order based on eigvals from large to small
                ii = np.argsort(eigvals)
                eigvals = eigvals[ii]
                eigvecs = eigvecs[:,ii]          

                attn_output = attn_weights @ value_vectors  

                EDs[hidx, alpidx, mr_ii] += (eigvals.sum())**2/(eigvals**2).sum() / N_batch
                """

                ###### 5. Plot results ######     

                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color
                colors = ['k', c_hyp]

                # Eigenvalues
                seq_len = K_tilde.shape[0]
                idxs = np.arange(1,seq_len + 1)
                idx_mid = len(idxs) // 2        

                lstyles = ['-', '--', '--']
                #axss[hidx][mr_ii].plot(idxs, eigvals, c=c_hyp, linestyle=lstyles[alphas.index(alpha)], label=r'$q$ kept')  # Non-uniform

                # plot attn-score colormap      
                ax_idx = mr_ii + alpidx * ncols
                print('\n')
                print(f'ax_idx = {ax_idx}')   
                #print(attn_weights)                
                print(attn_weights.shape)
                # K_tilde or attn_weights
                axss[hidx][ax_idx].imshow(K_tilde.detach().numpy(), cmap=cmap_attn)  # norm=cmap_norm                

        #break
    
        # attn-score and weights
        # for hidx in range(num_hidden_layers):
        #     for ii in range(pairwise_overlaps.shape[0]):
        #         axss[hidx][mr_ii + ncols*(ii + 1)].plot(np.arange(max_seq_len), pairwise_overlaps[ii,hidx].mean(0),
        #                                              c=colors[1])                

    # --------------------
    for hidx in range(num_hidden_layers):

        # share x, y axis for eigvecs (from second row of subfigures)
        share_xy_list = list(axss[hidx][nrows:])
        if len(share_xy_list) > 1:
            share_xy_list[0].get_shared_x_axes().join(share_xy_list[0], *share_xy_list)        

        for mr_ii in range(len(axss[hidx])):
            if mr_ii < ncols:
                #axss[hidx][mr_ii].set_xscale('log') 
                #axss[hidx][mr_ii].set_yscale('log')    
                pass    

            # ----- plot labels -----

            # subplot labels
            axss[hidx][mr_ii].text(
                0.0, 1.0, f'({ascii_lowercase[mr_ii]})', transform=(
                    axss[hidx][mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom', fontfamily='sans-serif')  # fontsize='medium',   

            # row labels 
            # if mr_ii % ncols == ncols - 1:
            #     axss[hidx][mr_ii].text(1.2, 0.5, row_labels[mr_ii//ncols], transform=(
            #                     axss[hidx][mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            #                     va='center', rotation='vertical')  # fontsize='medium',          

        
        # ----- FIG LABELS -----
        # col labels
        for col in range(ncols):
            bandwidth = epss[col]
            axss[hidx][col].set_title(rf'$\varepsilon = {{{bandwidth}}}$')   

        # axss[hidx][0].set_ylabel('Eigenspectrum')
        # axss[hidx][ncols].set_ylabel('Attn scores')
        # axss[hidx][ncols*2].set_ylabel('Attn weights')

        # for col in range(ncols):
        #     axss[hidx][col + ncols*(nrows-1)].set_xlabel('Token steps')
        # -----------------------

        # ----- SAVE FIGURE ------
        SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis')
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        layers, heads, hidden = config['num_hidden_layers'], config['num_attention_heads'], int(config['hidden_size'])
        #fig_file = 'fdm-'
        fig_file = 'attn_w-'
        fig_file += f'{model_name}-layers={layers}-heads={heads}-hidden={hidden}-l={hidx}'
        # if isfile(njoin(SAVE_DIR, fig_file)):
        #     version = len([fname for fname in os.listdir(SAVE_DIR) if fname==fig_file])
        #     fig_file += f'-v{version}'    
        fig_file += '.pdf'
        figs[hidx].savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')