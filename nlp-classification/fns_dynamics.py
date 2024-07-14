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
    dirname_idxs = []
    model_instances = []
    epss = []
    for dirname_idx, dirname in enumerate(dirnames):
        if f'-eps=' in dirname and 'opfns' in dirname:
            for s in dirname.split('-'):
                if 'eps' in s:
                    break
            eps = float(s[4:])
            if eps not in epss:
                epss.append(eps)
            for subdir in os.listdir(njoin(models_root, dirname)):
                if isfile(njoin(models_root, dirname, subdir, 'run_performance.csv')):
                    final_performance = pd.read_csv(njoin(models_root, dirname, subdir, 'final_performance.csv'))
                    dataset = final_performance.loc[0,'dataset_name']
                    print(f'Index {dirname_idx}: {dirname}')
                    dirname_idxs.append(dirname_idx)
                    model_instances.append(subdir)
                    break    
    assert len(dirname_idxs) <= len(dirnames), 'dirname_idxs cannot exceed dirnames'
    model_dirs = []
    for ii, dirname_idx in enumerate(dirname_idxs):
        model_dirs.append(njoin(models_root, dirnames[dirname_idx], model_instances[ii]))
    

    # Set up plots for later
    nrows, ncols = 2, len(epss)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=True,sharey=True)          
    if nrows == 1:
        if ncols > 1:
            axs = np.expand_dims(axs, axis=0)
        else:
            axs = np.expand_dims(axs, axis=[0,1])     
    axs = axs.flatten()    
      
    row_labels = ['Uniform', 'Non-uniform']
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
        assert 'op' in attn_setup['model_name'], 'Orthogonal projection matrix is required!'
        assert config['num_attention_heads'] == 1, 'Number of attention heads must be 1!'

        # ---------------------------------------- 1. Dataset setup ----------------------------------------
        if model_idx == 0:
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
            N_batch = 10
            dataset = attn_setup['dataset_name']
            #train_bs = int(train_setting.loc[0,'batch_size'])
            train_bs = 1
            eval_bs = 1    

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
         

        ###### 1. Compute geodesic distances and attention-score ######

        dist_mses = []
        bidxs = list(range(N_batch))
        batch_idx = train_dataset['attention_mask'].sum(-1).argmax().item()
        if batch_idx in bidxs:
            bidxs.remove(batch_idx)
        bidxs.append(batch_idx)

        for bidx in tqdm(bidxs):         

        ###### 2. Extract single data-point only ######
            X = train_dataset['input_ids'][bidx * train_bs : (bidx+1) * train_bs]  # batch_size must be one if jask is set to None
            Y = train_dataset['label'][bidx * train_bs : (bidx+1) * train_bs]
            bool_mask = train_dataset['attention_mask'][bidx * train_bs : (bidx+1) * train_bs]        
            if train_bs == 1:  # remove padding, only keep the seq from sos to eos        
                X_len = bool_mask.sum().item()
                X = X[:,:X_len]  
                attention_mask = None

            hidden_states = model.transformer.embeddings(X)
            if manifold == 'sphere':
                hidden_states = F.normalize(hidden_states,p=2,dim=-1)
            
            query_vectors = model.transformer.encoder.layer[0].attention.self.query(hidden_states)  # (N,B,HD)
            value_vectors = model.transformer.encoder.layer[0].attention.self.value(hidden_states)

            batch_size, seq_len, embed_dim = hidden_states.size()
            num_heads, head_dim = model.transformer.encoder.layer[0].attention.self.num_heads, model.transformer.encoder.layer[0].attention.self.head_dim                
            assert (
                embed_dim == model.transformer.encoder.layer[0].attention.self.embed_dim
            ), f"hidden_states should have embed_dim = {model.transformer.encoder.layer[0].attention.self.embed_dim}, but has {embed_dim}"        


            # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)                
            query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)       
            value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                                         

            # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
            if not config['qk_share']:
                key_vectors = model.transformer.encoder.layer[0].attention.self.key(hidden_states)    
                key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,H,N,D)   

                if manifold == 'rd':
                    Dist = torch.cdist(query_vectors, key_vectors, p=2)
                elif manifold == 'sphere':
                    eps = 1e-7  # for limiting the divergence from acos
                    query_vectors = F.normalize(query_vectors, p=2, dim=-1)
                    key_vectors = F.normalize(key_vectors, p=2, dim=-1)
                    Dist = torch.acos(torch.clamp(query_vectors @ key_vectors.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius
                    #Dist_true = torch.acos(hidden_states @ hidden_states.transpose(-2, -1)) * sphere_radius                
            else:
                if manifold == 'rd':
                    Dist = torch.cdist(query_vectors, query_vectors, p=2)     
                    Dist_true = torch.cdist(hidden_states, hidden_states, p=2)   
                elif manifold == 'sphere':
                    eps = 1e-7  # for limiting the divergence from acos
                    query_vectors = F.normalize(query_vectors, p=2, dim=-1)
                    Dist = torch.acos(torch.clamp(query_vectors @ query_vectors.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius                 
                    Dist_true = torch.acos(hidden_states @ hidden_states.transpose(-2, -1)) * sphere_radius      
                    Dist_true.masked_fill_(torch.eye(Dist_true.shape[1], Dist_true.shape[2])==1, 0)

                dist_mse = ((Dist[0,0] - Dist_true[0])**2).sum().detach().item() / Dist[0,0].numel()
                dist_mses.append(dist_mse)                      

            if attention_mask is not None:
                # type 1: key_pad_mask
                bool_mask = (attention_mask>=0).long()       
                attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,model.transformer.encoder.layer[0].attention.self.num_heads,1,-1])
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

        ###### 3. Extended analysis on removing non-uniform sampling (MC equiv to attn-score) ######

        a_ = 1
        if alpha != 2:
            attn_score_ = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))

            N_R_ = attn_score_.sum(-1)  # row sum
            N_C_ = attn_score_.sum(-2)  # col su                
            K_tilde = (N_R_**(-a_)).unsqueeze(-1) * attn_score_ * (N_C_**(-a_)).unsqueeze(-2)
        else:
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su              
            K_tilde = (N_R**(-a_)).unsqueeze(-1) * attn_score * (N_C**(-a_)).unsqueeze(-2)

        MC_ = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        eigvals_, eigvecs_ = torch.linalg.eigh(MC_)
        eigvals_ = eigvals_.detach().numpy()
        eigvecs_ = eigvecs_.detach().numpy()
        # order based on eigvals from small to large
        ii = np.argsort(eigvals_[0,0])
        eigvals_ = eigvals_[:,:,ii]
        eigvecs_ = eigvecs_[:,:,:,ii]

        ###### 4. Obtain attention weights based on model settings ######

        if a > 0:            
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)       

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive     

        WQ = model.transformer.encoder.layer[0].attention.self.query.weight
        WQ_eigvals, WQ_eigvecs = torch.linalg.eig(WQ)        
        WQ_eigvals = WQ_eigvals.detach().numpy()
        WQ_eigvecs = WQ_eigvecs.detach().numpy()

        # remove dropout for eval
        # attn_weights = F.dropout(attn_weights, p=model.transformer.encoder.layer[0].attention.self.dropout, 
        #                          training=model.transformer.encoder.layer[0].attention.self.training)           

        eigvals, eigvecs = torch.linalg.eigh(attn_weights)
        eigvals = eigvals.detach().numpy()
        eigvecs = eigvecs.detach().numpy()      
        # order based on eigvals from large to small
        ii = np.argsort(eigvals[0,0])
        eigvals = eigvals[:,:,ii]
        eigvecs = eigvecs[:,:,:,ii]          

        attn_output = attn_weights @ value_vectors  

        ###### 5. Plot results ######

        """
        # Projection matrix properties
        axs[model_idx,0].scatter(WQ_eigvals.real, WQ_eigvals.imag, s=0.75)        
        # Geodesic distance error
        dist_mses = np.array(dist_mses)
        density = gaussian_kde(dist_mses)
        xs = np.linspace(0,1,100)
        axs[model_idx,1].plot(xs,density(xs))    
        """        

        c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color

        # Eigenvalues
        idxs = np.arange(1,len(eigvals_[0,0])+1)
        idx_mid = len(idxs) // 2

        # Uniform
        axs[mr_ii].plot(idxs, eigvals_[0,0], c=c_hyp, linestyle='-', label=r'$q$ removed')
        # eye guide
        # power = alpha if alpha <= 2 else 2
        # eigvals_theory = idxs**power  
        # eigvals_theory = eigvals_theory / eigvals_theory[idx_mid]
        # eigvals_theory = eigvals_theory * eigvals[idx_mid] * 10
        # ax.plot(idxs, eigvals_theory, c=c_alpha, alpha=0.5, linewidth=1, linestyle='--')

        # Non-uniform
        axs[mr_ii + ncols].plot(idxs, eigvals[0,0], c=c_hyp, linestyle='-', label=r'$q$ kept')

        # Eigenvectors
        # Query/key projection onto lower-dim
        #dim1, dim2 = 0, 1
        #axs[model_idx,1].scatter(query_vectors[0,0,:,dim1].detach().numpy(), query_vectors[0,0,:,dim2].detach().numpy())
        #axs[mr_ii].scatter(eigvecs_[0,0,:,0], eigvecs_[0,0,:,1], c=c_hyp)

    for mr_ii in range(len(axs)):
        axs[mr_ii].set_xscale('log'); axs[mr_ii].set_yscale('log')        

        # ----- plot labels -----

        # subplot labels
        axs[mr_ii].text(
            0.0, 1.0, f'({ascii_lowercase[mr_ii]})', transform=(
                axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            va='bottom', fontfamily='sans-serif')  # fontsize='medium',   

        # if mr_ii // ncols == nrows-1:
        #     axs[mr_ii].set_xticks(epochs)
        #     axs[mr_ii].set_xticklabels(epochs)

        # row labels 
        if mr_ii % ncols == ncols - 1:
            axs[mr_ii].text(1.2, 0.5, row_labels[mr_ii//ncols], transform=(
                            axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                            va='center', rotation='vertical')  # fontsize='medium',          

    # col labels
    for col in range(ncols):
        bandwidth = epss[col]
        axs[col].set_title(rf'$\varepsilon = {{{bandwidth}}}$')     

    # --------------------

    FIGS_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
    layers, heads, hidden = config['num_hidden_layers'], config['num_attention_heads'], int(config['hidden_size'])
    fig_file = f'{model_name}-layers={layers}-heads={heads}-hidden={hidden}'
    # if isfile(njoin(FIGS_DIR, fig_file)):
    #     version = len([fname for fname in os.listdir(FIGS_DIR) if fname==fig_file])
    #     fig_file += f'-v{version}'
    fig_file += '-dm'
    fig_file += '.pdf'
    plt.savefig(njoin(FIGS_DIR, fig_file))            
    print(f'Figure saved in {njoin(FIGS_DIR, fig_file)}')