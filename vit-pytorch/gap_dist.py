import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import random
import torch
from time import time, sleep
from typing import Union
from tqdm import tqdm
#from constants import DROOT, MODEL_NAMES
from constants import *
from mutils import *

from os import makedirs
from os.path import isdir, isfile
from transformers.utils import logging

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings("ignore")    

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.cm import get_cmap

from string import ascii_lowercase
from torch.nn import functional as F  

#from data_utils import prepare_data
from data_utils import prepare_cifar10_data, prepare_mnist_data

from torchvision import transforms

# ----- global plot settings -----
c_clears = [0.5, 1]
markers = ['o', 'x']

cmap_attn = get_cmap('inferno')
cmap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
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
python -i spectral_gap.py\
 --models_root=.droot/1L-ps\=1-v4/config_qqv/mnist/layers\=1-heads\=1-hidden\=48-qqv/ --fns_type=oprdfnsvit --selected_alphas=1.2,2.0
"""

device = f'cuda' if torch.cuda.is_available() else "cpu"
ddp_world_size = 1

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='main_seq_classification.py training arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--models_root', default='', help='Pretrained models root')
    parser.add_argument('--fns_type', default='oprdfns'+MODEL_SUFFIX)
    #parser.add_argument('--N_batch', default=25, type=int)
    parser.add_argument('--selected_alphas', type=str, default='1.2,1.6,2.0')
    parser.add_argument('--is_attn_w', type=str2bool, nargs='?', default=True)

    parser.add_argument('--wandb_log', default=False, type=bool)

    args = parser.parse_args()    

    if not args.wandb_log:
        os.environ["WANDB_DISABLED"] = "true"

    repo_dir = os.getcwd()  # main dir 
    dev = torch.device(f"cuda:{torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")                          
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
    dirnames = sorted([dirname for dirname in os.listdir(models_root) if MODEL_SUFFIX in dirname])  

    # select based on bandwidth    
    DCT_ALL = collect_model_dirs(models_root, suffix=MODEL_SUFFIX)
    model_types = list(DCT_ALL.keys())

    #SELECTED_ALPHAS = [1.2,1.6,2.0]
    SELECTED_ALPHAS = [float(selected_alpha) for selected_alpha in str2ls(args.selected_alphas)]
    for model_type in model_types:
        if args.fns_type in model_type:
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            # ----- filter alphas -----
            if SELECTED_ALPHAS is not None:
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
    nrows, ncols = len(alphas), 2
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
      
    row_labels = ['Uniform', 'Non-uniform']
    for model_idx, model_dir in enumerate(model_dirs):        

        train_setting = pd.read_csv(njoin(model_dir, 'train_setting.csv'))
        if os.path.isdir(njoin(model_dir, 'run_performance.csv')):
            run_performance = pd.read_csv(njoin(model_dir, 'run_performance.csv'))
        train_setting = pd.read_csv(njoin(model_dir, 'train_setting.csv'))
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
        #mr_ii = epss.index(bandwidth)        

        print(f'alpha = {alpha}')
        print(config)

        # conditions for diffusion maps
        #assert config['qk_share'], 'QK weight-tying is required!'
        assert 'op' in attn_setup['model_name'], 'Orthogonal projection matrix is required!'
        assert config['num_attention_heads'] == 1, 'Number of attention heads must be 1!'

        # ----- load data -----
        dataset_name = attn_setup['dataset_name']
        batch_size = 1
        if model_idx == 0:
            # poor man's data loader
            if dataset_name.lower() == 'cifar10':
                #trainloader, testloader, _ = prepare_data(batch_size=batch_size, num_workers=ddp_world_size)    
                trainloader, testloader, _ = prepare_cifar10_data(batch_size=batch_size, num_workers=1)

                if trainloader.dataset[0][0].ndim == 2:
                    image_size = trainloader.dataset[0][0].shape[0]
                    num_channels = 1
                else:
                    image_size = trainloader.dataset[0][0].shape[1]
                    num_channels = trainloader.dataset[0][0].shape[0]

                num_classes = len(trainloader.dataset.classes)        

            elif dataset_name.lower() == 'mnist':
                #data_dir = njoin(DROOT,'DATA','cifar-10-batches-py')
                trainloader, testloader = prepare_mnist_data(batch_size=batch_size, num_workers=ddp_world_size)    

                if trainloader.dataset[0][0].ndim == 2:
                    image_size = trainloader.dataset[0][0].shape[0]
                    num_channels = 1
                else:
                    image_size = trainloader.dataset[0][0].shape[1]
                    num_channels = trainloader.dataset[0][0].shape[0]

                num_classes = len(trainloader.dataset.classes)                  

            elif dataset_name in ['pathfinder-classification', 'pathx-classification']:  # 'lra-cifar-classification'
                from lra_dataloading import Datasets

                # Get dataset creation function
                create_dataset_fn = Datasets[dataset_name]
                
                padded = False
                retrieval = False            
        
            # randomly select an image
            # ii = random.randint(0,len(trainloader.dataset))
            # X, Y = trainloader.dataset[ii]
            # if X.ndim==3:
            #     X = X[None]

        # ----- load model -----
        if 'rd' in args.fns_type:
            from vit_models.rdfns_vit import RDFNSViTForClassfication
            model = RDFNSViTForClassfication(config)                    
        elif 'sp' in args.fns_type:
            from vit_pytorch.spopfns_vit import SPOPFNSViTForClassfication
            model = SPOPFNSViTForClassfication(config)   
        else:
            continue
        checkpoint = njoin(model_dir, 'ckpt.pt')
        if isfile(checkpoint):            
            ckpt = torch.load(checkpoint, map_location=device)
            model.load_state_dict(ckpt['model'])
            model.eval()
        else:
            continue

        # selected data size
        N_data = 500
        gap_array = np.empty(N_data)
        classify_array = np.empty(N_data)
        for didx in tqdm(range(min(N_data, len(testloader.dataset)))):
            X, Y = testloader.dataset[didx]
            if X.ndim==3:
                X = X[None]            
            # get classification
            logits, _ = model(X)
            classify_array[didx] = int(logits.argmax(dim=1) == Y)

            x = model.embedding(X)
            # -----
            if config['is_preln']:
                x = model.encoder.blocks[0].layernorm_1(x)
            # Calculate the encoder's output
            #encoder_output, all_attentions = model.encoder(embedding_output, output_attentions=output_attentions)
            for lidx, block in enumerate(model.encoder.blocks):
                #x, attention_probs = block(x, output_attentions=output_attentions)
                # Self-attention
                # attention_output, attention_probs = \
                #     block.attention(x, output_attentions=output_attentions)   

                if args.fns_type[-8:] == 'spfnsvit':
                    x = F.normalize(x,p=2,dim=-1)
                #query = self.WQ(x)        
                query = x
                value = block.attention.WV(x)

                # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
                batch_size, sequence_length, _ = query.size()
                num_attention_heads, attention_head_size = block.attention.num_attention_heads, block.attention.attention_head_size

                alpha, bandwidth = block.attention.alpha, block.attention.bandwidth
                a = block.attention.a            
                d_intrinsic = attention_head_size

                query = query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
                value = value.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)

                if args.fns_type[-8:] == 'spfnsvit':
                    sphere_radius = block.attention.sphere_radius
                    eps = 1e-7  # for limiting the divergence from acos
                    if not block.attention.qk_share:        
                        key = block.attention.WK(x)
                        key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)                      
                        # geodesic distance on sphere      
                        # old method  
                        #g_dist = torch.acos(torch.clamp(query @ key.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius
                        # new method
                        g_dist = torch.acos_(query @ key.transpose(-2, -1)) * sphere_radius
                    else:
                        # geodesic distance on sphere      
                        # old method    
                        #g_dist = torch.acos(torch.clamp(query @ query.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius
                        # new method
                        dot = query @ query.transpose(-2, -1)
                        dot = dot.masked_fill_(torch.diag_embed(torch.ones(sequence_length, device=query.device))==1, 1)
                        g_dist = torch.acos_(dot) * sphere_radius           
                elif args.fns_type[-8:] == 'rdfnsvit':
                    if not block.attention.qk_share:        
                        key = block.attention.WK(x)
                        key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)                      
                        g_dist = torch.cdist(query, key, p=2)
                    else:
                        g_dist = torch.cdist(query, query, p=2)         

                    if config['is_rescale_dist']:
                                            
                        head_dim = attention_head_size
                        if block.attention.alpha >= 2:
                            dist_scale = (head_dim)**0.5
                        else:
                            dist_scale = head_dim**0.5 / (2**(1/head_dim) - 1)   
                        g_dist = g_dist / dist_scale
                
                # Calculate the attention scores
                if alpha < 2:
                    attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
                else:
                    attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))
                attn_score_shape = attn_score.shape
                if a > 0:
                    # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
                    N_R = attn_score.sum(-1)  # row sum
                    N_C = attn_score.sum(-2)  # col sum
                    K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)

                    attention_probs = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
                else:
                    attention_probs = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive

                # Calculate the attention output
                attention_output = attention_probs @ value
                # Resize the attention output
                # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
                # To (batch_size, sequence_length, all_head_size)
                attention_output = attention_output.transpose(1, 2) \
                                                .contiguous() \
                                                .view(batch_size, sequence_length, block.attention.all_head_size)
                # Project the attention output back to the hidden size
                attention_output = block.attention.output_projection(attention_output)
                #attention_output = block.attention.output_dropout(attention_output)

                ################################################################################

                # ----- pre-layernorm -----
                if config['is_preln']:
                    # Skip connection
                    x = x + attention_output
                    # Feed-forward network
                    mlp_output = block.mlp(block.layernorm_2(x))
                    # Skip connection
                    #x = x + mlp_output
                    encoder_output = x + mlp_output
                # ----- post-layernorm -----
                else:
                    # Skip connection + layernorm
                    x = block.layernorm_1(x + attention_output)                
                    # Feed-forward network
                    mlp_output = block.mlp(x)     
                    # Skip connection + layernorm
                    #x = block.layernorm_2(x + mlp_output)   
                    encoder_output = block.layernorm_2(x + mlp_output)

            # Calculate the logits, take the [CLS] token's output as features for classification
            logits = model.classifier(encoder_output[:, 0, :])

            # true output
            #logits_true = model(X)
        
            if a == 0:
                K_tilde = attn_score

            K_tilde = K_tilde[0,0]
            D_tilde = K_tilde.sum(-1)

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

            gap_array[didx] = eigvals[-1] - eigvals[-2]

        ###### 5. Plot results ######     

        alpidx = alphas.index(alpha)

        c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color
        colors = ['k', c_hyp]

        # ----- column 1 -----
        ax_idx = 0 + alpidx * ncols
        # attn-score        
        axss[lidx][ax_idx].hist(gap_array, color=c_hyp, bins=30, density=True)  # norm=cmap_norm               
        # ----- column 2 -----
        ax_idx = 1 + alpidx * ncols

        # Define colors for each class
        colors = np.where(classify_array == 1, 'tab:red', 'tab:blue')

        axss[lidx][ax_idx].scatter(gap_array, classify_array, c=colors, s=60, edgecolor='k')

        # ----- column 3 -----
        # ax_idx = 2 + alpidx * ncols   
        # eigvals_w, eigvecs_w = torch.linalg.eig(attention_probs[0,0])     
        # eigvals_w = eigvals_w.detach().numpy()
        # eigvecs_w = eigvecs_w.detach().numpy()  

        # ii = np.argsort(eigvals_w)
        # eigvals_w = eigvals_w[ii]
        # eigvecs_w = eigvecs_w[:,ii]         

        # axss[lidx][ax_idx].scatter(np.arange(1,len(eigvals_w)+1), eigvals_w,
        #                            c=c_hyp, alpha=1, s=4)

        print(f'{model_dir} done! \n')
    # --------------------
    for lidx in range(config['num_hidden_layers']):

        # share x, y axis for eigvecs (from second row of subfigures)
        # share_xy_list = list(axss[lidx][nrows:])
        # if len(share_xy_list) > 1:
        #     share_xy_list[0].get_shared_x_axes().join(share_xy_list[0], *share_xy_list)        

        for mr_ii in range(len(axss[lidx])):

            # ----- plot labels -----

            # subplot labels
            axss[lidx][mr_ii].text(
                0.0, 1.0, f'({ascii_lowercase[mr_ii]})', transform=(
                    axss[lidx][mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom', fontfamily='sans-serif')  # fontsize='medium',   

            # row labels 
            # if mr_ii % ncols == ncols - 1:
            #     axss[lidx][mr_ii].text(1.2, 0.5, row_labels[mr_ii//ncols], transform=(
            #                     axss[lidx][mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            #                     va='center', rotation='vertical')  # fontsize='medium',          
        
        # ----- FIG LABELS -----
        # col labels
        titles = ['Distribution', 'Classification']
        for col in range(ncols):
            #bandwidth = epss[col]
            #axss[lidx][col].set_title(rf'$\varepsilon = {{{bandwidth}}}$')   
            axss[lidx][col].set_title(titles[col])

        # row labels
        for row in range(nrows):
            alpha = alphas[row]
            axss[lidx][row*ncols].set_ylabel(rf'$\alpha = {{{alpha}}}$')

        # ----- SAVE FIGURE ------
        SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis')
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        layers, heads, hidden = config['num_hidden_layers'], config['num_attention_heads'], int(config['hidden_size'])
        model_name = attn_setup['model_name']
        fig_file = args.models_root.split('/')[1] + '-'
        fig_file += 'gap_dist-'        
        fig_file += f'{model_name}-ds={dataset_name}-L={layers}-H={heads}-D={hidden}-l={lidx}'
        if config['qk_share']:
            fig_file += '-qqv'
        else:
            fig_file += '-qkv'  
        fig_file += '.pdf'
        plt.tight_layout(rect=[0, 0, 0.93, 1])
        figs[lidx].savefig(njoin(SAVE_DIR, fig_file))            
        #plt.show()
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')    