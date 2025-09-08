import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import torch
from time import time
from torch.nn import functional as F  
from tqdm import tqdm
from os import makedirs
from os.path import isdir

from constants import HYP_CMAP, HYP_CNORM, FIGS_DIR, MODEL_SUFFIX
from UTILS.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from UTILS.mutils import dijkstra_matrix, fdm_kernel
from models.rdfnsformer import RDFNSformer
from models.dpformer import DPformer

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from UTILS.data_utils import glove_create_examples
from UTILS.dataloader import load_dataset_and_tokenizer
from UTILS.figure_utils import matrixify_axs, label_axs

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase

matplotlib.use("Agg")

# ----- Global Variables -----
MARKERSIZE = 4
BIGGER_SIZE = 10
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def count_trailing_zeros(tensor):
    # Reverse the tensor and find the first non-zero element
    reversed_tensor = torch.flip(tensor, dims=[0])
    nonzero_indices = torch.nonzero(reversed_tensor, as_tuple=True)[0]
    
    if len(nonzero_indices) == 0:  # All elements are zero
        return len(tensor)
    else:  # Count zeros from the end
        return nonzero_indices[0].item()

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/fdm.py arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--models_root', default='', help='Pretrained models root')
    parser.add_argument('--fns_type', default='oprdfnsformer')  # 'spopfns'+MODEL_SUFFIX
    parser.add_argument('--is_include_pad_mask', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--is_3d', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_log', default=False, type=bool)

    args = parser.parse_args()   

    t0 = time()
    # ----- set up -----
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    print(f'device: {device} \n')

    # Get model setting from dir
    models_root = args.models_root.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(models_root) if 'former' in dirname])  

    # select based on bandwidth    
    DCT_ALL = collect_model_dirs(models_root, suffix=MODEL_SUFFIX)
    model_types = list(DCT_ALL.keys())
    dp_types = [model_type for model_type in model_types if 'dp' in model_type]

    #SELECTED_ALPHAS = None
    SELECTED_ALPHAS = [1.2, 1.6, 2]
    SELECTED_EPSS = [1]
    for model_type in model_types:
        if args.fns_type in model_type:            
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            if SELECTED_ALPHAS is not None:
                # ----- filter alphas -----
                df_model = df_model[df_model['alpha'].isin(SELECTED_ALPHAS)]
                # ------------------------
                df_model.reset_index(drop=True, inplace=True)
            #if EXCLUDED_EPSS is not None:
            if SELECTED_EPSS is not None:
                # ----- filter bandwidth -----
                #df_model = df_model[~df_model['bandwidth'].isin(EXCLUDED_EPSS)]
                df_model = df_model[df_model['bandwidth'].isin(SELECTED_EPSS)]
                # ------------------------
                df_model.reset_index(drop=True, inplace=True)
            break

    # ----- general settings -----
    # num_attention_heads, num_hidden_layers, hidden_size = df_modedm_l.loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]
    # dataset = df_model.loc[0,'dataset_name']

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())  # small to large
    epss = sorted(df_model.loc[:,'bandwidth'].unique())  

    #nrows, ncols = 1, config['n_layers']
    #nrows, ncols = len(epss), len(alphas)
    if 'config_qqv' in args.models_root:
        nrows, ncols = 7, len(alphas)
    else:
        nrows, ncols = 3, len(alphas)

    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=False,sharey=False)  
    axs = matrixify_axs(axs, nrows, ncols)  # make axs 2D array
    label_axs(fig, axs)                     # alphabetically label figures

    # Create figure and GridSpec layout
    # fig = plt.figure(figsize=(12, 6))  
    # gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])  # 3 columns, last one is wider
    # # Create 2×2 subfigures (small ones)
    # ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    # ax2 = fig.add_subplot(gs[0, 1])  # Top-right
    # ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left
    # ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right
    # axs = np.array([[ax1, ax2], [ax3, ax4]])
    # # Create large subfigure (taking up 2 rows)
    # ax5 = fig.add_subplot(gs[:, 2])  # Spans both rows (large plot)

    t1 = time()
    #print(f'Time 1: {t1 - t0}s')

    for eps_idx, eps in enumerate(epss):

        model_dirs = []
        for ii in range(df_model.shape[0]):
            ensembles = df_model.loc[ii,'ensembles']            
            if ensembles > 0 and df_model.loc[ii,'bandwidth'] == eps:
                seed = str(sorted(df_model.loc[ii,'seeds'])[-1])
                model_dirs.append(njoin(df_model.loc[ii,'model_dir'], f'model={seed}'))

                #attn_setup, config, _, _ = load_model_files(model_dir)
        print(f'Bandwidth: {eps}, seed selected: {seed}')

        model_dirs = sorted(model_dirs)
        g_distss, attn_scoress, attention_weightss = [], [], []
        for model_idx, model_dir in enumerate(model_dirs):

            # ----- load pretrained_model -----    
            attn_setup, config, run_performance, train_setting = load_model_files(model_dir)

            fix_embed = attn_setup['fix_embed']
            pretrained_model_name = config['pretrained_model_name'] if fix_embed else False

            config['dataset_name'] = attn_setup['dataset_name']
            config['max_len'] = config['seq_len']
            manifold = attn_setup['manifold']      

            model = RDFNSformer(config, is_return_dist=True)     
            model = model.to(device)
            checkpoint = njoin(model_dir, 'ckpt.pt')  # assuming model's is trained
            ckpt = torch.load(checkpoint, map_location=torch.device(device))
            model.load_state_dict(ckpt['model'])
            model.eval()

            if model_idx == 0:  
                main_args = AttrDict(config)
                #batch_size = int(train_setting.loc[0,'batch_size'])
                batch_size = 1                    
                # ---------------------------------------- poor man's data loader ----------------------------------------
                # tokenizer_name is hard set in main.py for the following case
                if main_args.dataset_name == 'imdb' and not main_args.fix_embed:
                    main_args.tokenizer_name = 'sentencepiece'  
                    main_args.pretrained_model = 'wiki.model'
                    main_args.vocab_file = 'wiki.vocab'
                tokenizer, train_loader, test_loader, train_size, eval_size, steps_per_epoch, num_classes =\
                    load_dataset_and_tokenizer(main_args, batch_size)          
                # --------------------------------------------------------------------------------------------------------                          

                t2 = time()
                #print(f'Time 2: {t2 - t1}s')

                # record length of sequences
                X_lens = []
                for ii in tqdm(range(len(train_loader.dataset))):
                    X, Y = train_loader.dataset[ii]
                    X_lens.append(config['max_len'] - count_trailing_zeros(X) )
                max_len_idxs = np.where(np.array(X_lens) == config['max_len'])[0]
                idx = 0
                X, Y = train_loader.dataset[max_len_idxs[idx]]
                X_len = config['max_len'] - count_trailing_zeros(X)
                print(f'Data sequence length: {X_len}')

            # ----- fdm -----
            alpha, bandwidth, a = attn_setup['alpha'], attn_setup['bandwidth'], attn_setup['a']
            d_intrinsic = attn_setup['d_intrinsic'] if alpha < 2 else None

            outputs, attention_weights, g_dists = model(X[None].to(device))       
             
            t3 = time()
            #print(f'Time 3: {t3 - t2}s')

            attention_scores = []            
            for lidx, g_dist in enumerate(g_dists):

                # --------------------------------------------------------------------------------
                pad_id = 0
                positions = torch.arange(X[None].size(1), device=X[None].device, dtype=X[None].dtype).repeat(X[None].size(0), 1) + 1
                position_pad_mask = X[None].eq(pad_id)
                positions.masked_fill_(position_pad_mask, 0)

                embeddings = model.embedding(X[None]) + model.pos_embedding(positions)

                # PCA            
                n_components = 2
                pca = PCA(n_components=n_components, random_state=int(seed))
                #pca.fit(W_word.detach().numpy())
                # pca_results = pca.fit_transform(W_word.detach().numpy())
                # data = StandardScaler().fit_transform(embeddings.detach().numpy()) # Normalise    

                data = embeddings.detach().numpy() # No normalise
                ##### Q = K case #####
                if not config['qk_share']:
                    #Q = model.layers[0].mha.fns_attn.WQ(data[0])
                    Q = data[0]
                    pca_results = pca.fit_transform(Q)
                    Q_x, Q_y = pca_results.T  

                    K = model.layers[0].mha.WK(torch.tensor(data[0])).detach()
                    pca_results = pca.fit_transform(K)
                    K_x, K_y = pca_results.T  
                else:       
                    pca_results = pca.fit_transform(data[0])
                    x, y = pca_results.T                                                                                            

                # plot based on embeddings
                if fix_embed:
                    if pretrained_model_name == 'glove':
                        rad = 6.1  # glove                        
                    elif pretrained_model_name == 'distilbert-base-uncased':
                        rad = 1  # distil-bert-uncased
                    elif pretrained_model_name == 'albert-base-v2':
                        rad = 0.5  # albert-base-v2

                    # text
                    if pretrained_model_name == 'glove':
                        txts = [glove.itos[X[ii]] for ii in range(len(X))]
                    elif pretrained_model_name in ['distilbert-base-uncased', 'albert-base-v2']:
                        txts = tokenizer.decode(X)

                else:
                    rad = 5.95
                    txts = tokenizer.convert_ids_to_tokens(list(X[None][0].detach().numpy()))

                center = data[0].mean(0)
                far_xy_idxs, close_xy_idxs = [], []
                for ii, embd_coordinates in enumerate(data[0]):
                    if ((embd_coordinates - center)**2).sum() >= rad**2:
                        far_xy_idxs.append(ii)
                    else:
                        close_xy_idxs.append(ii)
                close_xy_idxs = np.array(close_xy_idxs)
                far_xy_idxs = np.array(far_xy_idxs)                    
                # --------------------------------------------------------------------------------

                if args.is_include_pad_mask:
                    g_dist = g_dist[:,:,:X_len,:X_len]
                else:
                    g_dist = g_dist[:,:,:,:]
                    X_len = config['max_len']                
                if 'v2_' not in manifold:
                    attn_score = dist_to_score(g_dist, alpha, bandwidth, d_intrinsic=d_intrinsic)
                else:
                    attn_score = dist_to_score(g_dist, alpha, bandwidth, d_intrinsic=1)
                attention_scores.append(attn_score)

                # determine furthest distance and coordinates in embedding space
                max_g_dist = g_dist.max().item()
                m = g_dist.view(1, -1).argmax(1)
                max_g_dist_indices = torch.cat(((m // X_len).view(-1, 1), (m % X_len).view(-1, 1)), dim=1)[0]          


                if a > 0:            
                    N_R = attn_score.sum(-1)  # row sum
                    N_C = attn_score.sum(-2)  # col sum                
                    K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)       
                else:
                    K_tilde = attn_score

                K_tilde = K_tilde[0,0]
                D_tilde = K_tilde.sum(-1)

                attention_weight = F.normalize(K_tilde,p=1,dim=-1)  # can do this as the attn weights are always positive

                min_attn_w, max_attn_w = attention_weight.min(), attention_weight.max()
                median_attn_w = torch.median(attention_weight).item()

                # Dijkstra to determine the shortest path between tokens furthest apart
                shortest_distance, shortest_path = dijkstra_matrix(-np.log(attention_weight.detach().numpy()), 
                                                                   max_g_dist_indices[0].item(), 
                                                                   max_g_dist_indices[1].item())

                print(f'alpha: {alpha}')
                print(f'g_dist min: {g_dist.min()}, g_dist max: {g_dist.max()}')
                print(f'attn_score min: {attn_score[0,0].min()}, max: {attn_score[0,0].max()}')
                print(f'attention_weight min: {min_attn_w}, max: {max_attn_w}, median: {median_attn_w}')
                print('\n')
                print(f'most likely path: {shortest_path}')        
                print(f'probability: {np.exp(-shortest_distance)}')    

                # ----- compute spectrum of MC -----
                ##### Q = K case #####
                if config['qk_share']:
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
                ##### Q != K case #####
                else:
                    # frequency (need to tune later)
                    #q = 1/3
                    q = 1/4*((attn_score[0,0] - attn_score[0,0].T).abs().max().item())  # q
                    # Magnetic transform
                    K_S = (attn_score[0,0] + attn_score[0,0].T).detach()/2  # symmetrized
                    K_A = -(attn_score[0,0] - attn_score[0,0].T).detach()   # anti-symmetrized
                    D = torch.diag(K_S.sum(-1))
                    D_inv = torch.diag(1/K_S.sum(-1))
                    arg = 2j*np.pi*q*K_A.numpy()
                    H_q = (D_inv**(0.5)).to(torch.complex64) @ (K_S * torch.exp(torch.tensor(arg, dtype=torch.complex64))) @ (D_inv**(0.5)).to(torch.complex64)
                    H_q = 0.5 * (H_q + H_q.mH)
                    H_q = H_q.real
                    #H_q = scipy.linalg.fractional_matrix_power(m, T)  # here T is a terminal time component 
                    #assert (H_q.numel()==(H_q==H_q.T).sum()), 'H_q is asymmetric'

                    eigvals, eigvecs = torch.linalg.eigh(H_q)

                # ----------------------------------

                #attn_output = attention_weight @ value_vectors  

                ###### 5. Plot results ######     

                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color
                colors = ['k', c_hyp]                

                if pretrained_model_name == 'glove':
                    #thresh = 5e-17
                    thresh = 5e-17
                else:
                    thresh = 1e-16
                #thresh = 1e-5
                scale = 5e-4
                #scale = 0.25                                                        

                # diffusion map embeddings
                #axs[1,alphas.index(alpha)].scatter(eigvecs[far_xy_idxs,-1], eigvecs[far_xy_idxs,-2], c=c_hyp, s=1.5)

                N_plot_tokens = 15

                # PCA embeddings         
                if not config['qk_share']:
                    axs[0,alphas.index(alpha)].scatter(Q_x[:N_plot_tokens], Q_y[:N_plot_tokens], 
                                                       label='Queries', lw=.5, c='#dd1c77', edgecolors="k", s=20, zorder=2)   
                 
                    axs[0,alphas.index(alpha)].scatter(K_x[:N_plot_tokens], K_y[:N_plot_tokens], 
                                                       label='Keys', lw=.5, c='#a8ddb5', edgecolors="k", s=20, zorder=2)   


                    # for i in range(X_len):
                    #     for j in range(X_len):             
                    for i in range(N_plot_tokens):
                        for j in range(N_plot_tokens):  
                            if attention_weight[i,j] > thresh and i!=j and [i,j] :
                                axs[0,alphas.index(alpha)].plot([Q_x[i], K_x[j]], [Q_y[i], K_y[j]], 
                                        c=c_hyp, linewidth=scale * attention_weight[i,j].item(), #linewidth=scale * 1/(0.2-np.log(1 + attention_weight[i,j].item())), 
                                        linestyle='-')  # linewidth=scale * attention_weight[i,j].item(),
                                        #zorder=1)   

                    # asymmetric diffusion map embeddings
                    axs[1,alphas.index(alpha)].scatter(eigvecs[:N_plot_tokens,-1], eigvecs[:N_plot_tokens,-2], 
                                                       c=c_hyp, marker='x', alpha=0.75, s=4)    
                    
                    # label text                
                    for ii, idx in enumerate(range(N_plot_tokens)):
                        txt = txts[ii]
                        axs[1,alphas.index(alpha)].annotate(txt, (eigvecs[idx,-1], eigvecs[idx,-2]), 
                                                            size=5.5, zorder=3)
                    
                    # eigenvalues of magnetic transform H_q
                    axs[2,alphas.index(alpha)].scatter(np.arange(1,len(eigvals)+1), eigvals.numpy()[::-1], c=c_hyp, alpha=1, s=4)   
                    #axs[2,alphas.index(alpha)].set_xscale('log'); axs[2,alphas.index(alpha)].set_yscale('log')

                else:    

                    # shortest path 
                    # for idx in range(len(shortest_path[:-1])):
                    #     i, j = shortest_path[idx:idx+2]
                    #     axs[0,alphas.index(alpha)].plot([x[i], x[j]], [y[i], y[j]], 
                    #             c='grey', alpha=0.6, linewidth=10*scale * attention_weight[i,j].item(), 
                    #             linestyle='--')                       

                    # for i in range(X_len):               ##### if you want all tokens #####
                    #     for j in range(X_len):
                    for i in far_xy_idxs:                  ###### if you want tokens that are 'further' apart ######
                        for j in far_xy_idxs:                
                            # attention score
                            # if attn_score[0,0,i,j] > thresh and i!=j:
                            #     axs[0,alphas.index(alpha)].plot([x[i], x[j]], [y[i], y[j]], 
                            #             c=c_hyp, linewidth=scale * attn_score[0,0,i,j].item(), 
                            #             linestyle='-')
                            #             #zorder=1)              

                            # 1 / [1 - log(attention_weights)]
                            if attention_weight[i,j] > thresh and i!=j and [i,j] :
                                axs[0,alphas.index(alpha)].plot([x[i], x[j]], [y[i], y[j]], 
                                        c=c_hyp, linewidth=scale * attention_weight[i,j].item(), #linewidth=scale * 1/(0.2-np.log(1 + attention_weight[i,j].item())), 
                                        linestyle='-')  # linewidth=scale * attention_weight[i,j].item(),
                                        #zorder=1)   

                    axs[0,alphas.index(alpha)].scatter(x[far_xy_idxs], y[far_xy_idxs], 
                                                        c='k', marker='.', alpha=1, s=1)     
                    # Furtherest tokens in PCA space
                    axs[0,alphas.index(alpha)].scatter(x[max_g_dist_indices[0]], y[max_g_dist_indices[0]], 
                                                        c='r', marker='x', alpha=0.75, s=4)   
                    axs[0,alphas.index(alpha)].scatter(x[max_g_dist_indices[1]], y[max_g_dist_indices[1]], 
                                                        c='r', marker='x', alpha=0.75, s=4)  
                    # Furtherest tokens in diffusion map space
                    axs[1,alphas.index(alpha)].scatter(eigvecs[max_g_dist_indices[0],-1], eigvecs[max_g_dist_indices[0],-2], 
                                                        c='r', marker='x', alpha=0.75, s=4)   
                    axs[1,alphas.index(alpha)].scatter(eigvecs[max_g_dist_indices[1],-1], eigvecs[max_g_dist_indices[1],-2], 
                                                        c='r', marker='x', alpha=0.75, s=4)                  

                    # plot most probable path between distant tokens
                    for idx in shortest_path[1:-1]:
                        if idx not in far_xy_idxs:
                            axs[0,alphas.index(alpha)].scatter(x[idx], y[idx], 
                                                            c='green', marker='o', alpha=1, s=1.5)

                            # axs[alphas.index(alpha), 1].scatter(eigvecs[idx,-1], eigvecs[idx,-2], 
                            #                                  c='green', marker='o', alpha=1, s=1.5)

                    scale2 = 2 * scale
                    for ii in range(len(shortest_path)-1):
                        i, j = shortest_path[ii], shortest_path[ii+1]
                        axs[0,alphas.index(alpha)].plot([x[i], x[j]], [y[i], y[j]], 
                                                        alpha=0.75,
                                                        c='grey', linewidth=scale * attention_weight[i,j].item(), #linewidth=scale2 * 1/(0.2-np.log(1 + attention_weight[i,j].item())), 
                                                        linestyle=':')
                        # axs[alphas.index(alpha), 1].plot([eigvecs[i,-1], eigvecs[j,-1]], [eigvecs[i,-2], eigvecs[j,-2]],
                        #          alpha=0.75, c='grey', linestyle='--')

                    # attention weights vs distance
                    markers = ['.', '^', 's']
                    linestyles = ['-', '--', '-.']
                    trans = [1, 0.75,0.5]
                    #q_rows = [0,1,2]
                    q_rows = [0, 100, 400]
                    if alphas.index(alpha) == 0:
                        if len(dp_types) > 0:
                            df_model_other = DCT_ALL[dp_types[0]]
                            model_dir_other = njoin(df_model.loc[ii,'model_dir'], f'model={seed}')

                            attn_setup_other, config_other, run_performance_other, train_setting_other =\
                                load_model_files(model_dir_other)

                            config_other['dataset_name'] = attn_setup_other['dataset_name']
                            if config_other['dataset_name'] == 'imdb' and 'num_classes' not in config_other.keys():
                                config_other['num_classes'] = 2
                            config_other['max_len'] = config_other['seq_len']

                            model_other = DPformer(config_other, is_return_dist=True)  
                            model_other = model_other.to(device)   
                            checkpoint_other = njoin(model_dir_other, 'ckpt.pt')
                            ckpt_other = torch.load(checkpoint_other, map_location=torch.device(device))
                            model_other.load_state_dict(ckpt_other['model'])
                            model_other.eval()    
                            outputs_other, attention_weights_other, g_dists_other = model_other(X[None].to(device))   
                            g_dist_other = g_dists_other[0]              
                            attention_weight_other = attention_weights_other[0]

                            for q_ii, q_row in enumerate(q_rows):
                                axs[2,alphas.index(alpha)].scatter(g_dist_other[0,0,q_row,:].detach().flatten().numpy(), 
                                                                attention_weight_other[0,0,q_row,:].detach().flatten().numpy(), 
                                                                c='k', s=2, marker=markers[q_ii],
                                                                alpha=trans[q_ii])
                                
                            axs[6,alphas.index(alpha)].hist(attention_weight_other.detach().flatten().numpy(), 750,
                                                            color='k', alpha=0.5, density=True)                            

                    # attention weights
                    for q_ii, q_row in enumerate(q_rows):
                        axs[2,alphas.index(alpha)].scatter(g_dist[0,0,q_row,:].detach().flatten().numpy(), 
                                                        attention_weight[q_row,:].detach().flatten().numpy(), 
                                                        c=c_hyp, s=2, marker=markers[q_ii],
                                                        alpha=trans[q_ii])

                    dists = np.arange(1, g_dist.max().item(), 1)
                    if alpha < 2:
                        if 'v2_rd' not in args.fns_type:
                            d_intrinsic_kernel = config['d_intrinsic']
                        else:
                            d_intrinsic_kernel = 1

                    # extract from model
                    if config['is_rescale_dist']:
                        dist_scale = model.layers[0].mha.fns_attn.dist_scale
                    else:
                        dist_scale = 1

                    kernel_values = fdm_kernel(dists, alpha, d_intrinsic_kernel, 
                                            bandwidth=config['bandwidth'], 
                                            dist_scale=dist_scale)
                    axs[2,alphas.index(alpha)].plot(dists, kernel_values, c=c_hyp)

                    # index vs distance
                    for q_ii, q_row in enumerate(q_rows[:1]):            
                        axs[3,alphas.index(alpha)].plot(list(range(1,config['max_len']+1)), 
                                                        g_dist[0,0,q_row,:].detach().flatten().numpy(),                                                      
                                                        c=c_hyp, linestyle=linestyles[q_ii],
                                                        linewidth=1,
                                                        alpha=trans[q_ii])   

                    # distance histogram
                    axs[4,alphas.index(alpha)].hist(g_dist[0,0,:,:].detach().flatten().numpy(), 750,
                                                    color=c_hyp, density=True)

                    # attention score histogram
                    axs[5,alphas.index(alpha)].hist(attn_score[0,0,:,:].detach().flatten().numpy(), 750,
                                                    color=c_hyp, density=True)

                    # attention weight histogram
                    axs[6,alphas.index(alpha)].hist(attention_weight.detach().flatten().numpy(), 750,
                                                    color=c_hyp, density=True)

                    axs[2,alphas.index(alpha)].set_xscale('log')
                    axs[2,alphas.index(alpha)].set_yscale('log')

                    # label text                
                    for ii, idx in enumerate(far_xy_idxs):
                        txt = txts[ii]
                        axs[0,alphas.index(alpha)].annotate(txt, (x[idx], y[idx]), 
                                                            size=5.5, zorder=3)
                        axs[1,alphas.index(alpha)].annotate(txt, (eigvecs[idx,-1], eigvecs[idx,-2]), 
                                                            size=5.5, zorder=3)

                    # label rows
                    axs[0,alphas.index(alpha)].set_title(rf'$\alpha$ = {alpha}', fontsize=10)    
                    axs[0,alphas.index(alpha)].set_xlabel(rf'PC$_1$',fontsize=8)
                    axs[0,alphas.index(alpha)].set_ylabel(rf'PC$_2$',fontsize=8)
                    axs[1,alphas.index(alpha)].set_xlabel(rf'DM$_1$',fontsize=8)
                    axs[1,alphas.index(alpha)].set_ylabel(rf'DM$_2$',fontsize=8)
                    axs[2,alphas.index(alpha)].set_xlabel(rf'Distance',fontsize=8)
                    axs[2,alphas.index(alpha)].set_ylabel(rf'Attn Weights',fontsize=8)   
                    axs[3,alphas.index(alpha)].set_xlabel(rf'Index',fontsize=8)
                    axs[3,alphas.index(alpha)].set_ylabel(rf'Distance',fontsize=8)                     
                    axs[4,alphas.index(alpha)].set_xlabel(rf'Distance',fontsize=8)
                    axs[4,alphas.index(alpha)].set_ylabel(rf'Density',fontsize=8)                                
                    axs[5,alphas.index(alpha)].set_xlabel(rf'Attn Score',fontsize=8)
                    axs[5,alphas.index(alpha)].set_ylabel(rf'Density',fontsize=8) 
                    axs[6,alphas.index(alpha)].set_xlabel(rf'Attn Weights',fontsize=8)
                    axs[6,alphas.index(alpha)].set_ylabel(rf'Density',fontsize=8)                 

            g_distss.append(g_dists)
            attn_scoress.append(attention_scores)
            attention_weightss.append(attention_weights)

        t4 = time()
        #print(f'Time 4: {t4 - t3}s')

        if config['qk_share']:
            for row in range(4,7):
                for col in range(len(alphas)):
                    axs[row,col].set_xscale('log'); axs[row,col].set_yscale('log')

            # axs[0,0].set_title('Embedding PCs')
            # axs[0,1].set_title('Diffusion map')
            axs[0,1].text(1.1, 0.5, 'Embedding PCs', transform=axs[0,1].transAxes,
            rotation=90, va='center', ha='center', fontsize=10)
            axs[1,1].text(1.1, 0.5, 'Diffusion map', transform=axs[1,1].transAxes,
            rotation=90, va='center', ha='center', fontsize=10)        
            # axs[2,1].text(1.1, 0.5, 'Attention weights', transform=axs[1,1].transAxes,
            # rotation=90, va='center', ha='center', fontsize=10)           

            # legends
            # axs[0,0].plot([], [], alpha=0.75, c='grey',linestyle=':', label='Shortest path')
            # axs[0,0].plot([], [], alpha=1, c='k',linestyle='-', label='Attention weights')
            # axs[0,0].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(-0.05, 1.3))        

        if 'L-hidden' in args.models_root.split('/')[1]:
            SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1])
        else:
            SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1], 
                            args.models_root.split('/')[2])
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)       
        qk_affix = 'qqv' if config['qk_share'] else 'qkv'
        fig_file = f'fdm-{args.fns_type}-eps={eps}-pretrained_embd={pretrained_model_name}-{qk_affix}.pdf'
        plt.tight_layout()
        plt.savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')    

        t5 = time()
        #print(f'Time 5: {t5 - t4}s')