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
# from sklearn.decomposition import PCA   
# from sklearn.preprocessing import StandardScaler
from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase
from torchtext.vocab import GloVe
from constants import HYP_CMAP, HYP_CNORM, FIGS_DIR, MODEL_SUFFIX
from UTILS.data_utils import glove_create_examples
from UTILS.dataloader import load_dataset_and_tokenizer
from UTILS.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from UTILS.mutils import dijkstra_matrix, fdm_kernel
from models.model import Transformer
import networkx as nx

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
    parser = argparse.ArgumentParser(description='nlp-tutorial/attn_graph_final.py arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--model_dir', default='', help='Pretrained model directory')
    parser.add_argument('--bdwth', type=float, default=1, help='Executed bandwidth for DM')
    parser.add_argument('--is_include_pad_mask', type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()   

    # ----- set up -----
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    print(f'device: {device} \n')

    model_dir = args.model_dir

    # ----- load pretrained_model -----    
    attn_setup, config, run_performance, train_setting = load_model_files(model_dir)

    fix_embed = attn_setup['fix_embed']
    pretrained_model_name = config['pretrained_model_name'] if fix_embed else False

    config['dataset_name'] = attn_setup['dataset_name']
    config['max_len'] = config['seq_len']    
    # correction for config model_name
    if 'model_name' not in config.keys():
        config['model_name'] = attn_setup['model_name'][2:] if config['is_op'] else attn_setup['model_name']  

    # -------------------------------------------------------------------------
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

    # record length of sequences
    X_lens = []
    for ii in tqdm(range(len(train_loader.dataset))):
        X, Y = train_loader.dataset[ii]
        X_lens.append(config['max_len'] - count_trailing_zeros(X) )
    # target_len_idxs = np.where(np.array(X_lens) == config['max_len'])[0]
    target_len_idxs = np.where(np.array(X_lens) == 500)[0]
    idx = 0
    X, Y = train_loader.dataset[target_len_idxs[idx]]
    X_len = config['max_len'] - count_trailing_zeros(X)
    print(f'Data sequence length: {X_len}')

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
    # -------------------------------------------------------------------------

    # ---------- FNS ----------
    if config['model_name'][-9:] == 'fnsformer':

        print('Running fnsformer. \n')

        manifold = attn_setup['manifold']  
        # correction for alpha
        if 'alpha' not in config.keys():
            config['alpha'] = attn_setup['alpha']

        model = Transformer(config,  is_return_dist = True)   
        model = model.to(device)
        checkpoint = njoin(model_dir, 'ckpt.pt')  # assuming model's is trained
        ckpt = torch.load(checkpoint, map_location=torch.device(device))
        model.load_state_dict(ckpt['model'])
        model.eval()

        # implemented bandwidth
        bdwth = args.bdwth

        # ----- fdm -----
        alpha, bandwidth, a = attn_setup['alpha'], attn_setup['bandwidth'], attn_setup['a']
        d_intrinsic = attn_setup['d_intrinsic'] if alpha < 2 else None

        outputs, attention_weights, g_dists = model(X[None].to(device))             
        for lidx, g_dist in enumerate(g_dists):

            # --------------------------------------------------------------------------------
            pad_id = 0
            positions = torch.arange(X[None].size(1), device=X[None].device, dtype=X[None].dtype).repeat(X[None].size(0), 1) + 1
            position_pad_mask = X[None].eq(pad_id)
            positions.masked_fill_(position_pad_mask, 0)

            embeddings = model.embedding(X[None].to(device)).to(device) + model.pos_embedding(positions.to(device)).to(device)

            # PCA            
            n_components = 2
            seed = attn_setup['seed']
            # pca = PCA(n_components=n_components, random_state=int(seed))
            #pca.fit(W_word.detach().numpy())
            # pca_results = pca.fit_transform(W_word.detach().numpy())
            # data = StandardScaler().fit_transform(embeddings.detach().numpy()) # Normalise    

            if 'cuda' in device:
                data = embeddings.cpu().detach().numpy() # No normalise
            else:
                data = embeddings.detach().numpy() # No normalise
            ##### Q = K case #####
            # if not config['qk_share']:
            #     #Q = model.layers[0].mha.fns_attn.WQ(data[0])
            #     Q = data[0]
            #     pca_results = pca.fit_transform(Q)
            #     Q_x, Q_y = pca_results.T  

            #     K = model.layers[0].mha.WK(torch.tensor(data[0])).detach()
            #     pca_results = pca.fit_transform(K)
            #     K_x, K_y = pca_results.T  
            # else:       
            #     pca_results = pca.fit_transform(data[0])
            #     x, y = pca_results.T                                                                                            

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

            if X_len < config['max_len'] and not args.is_include_pad_mask:
                g_dist = g_dist[:,:,:X_len,:X_len]
            else:
                g_dist = g_dist[:,:,:,:]
                X_len = config['max_len']      

            # cancel distance rescaling if used
            # if config['is_rescale_dist']:
            #     g_dist = g_dist * model.layers[lidx].mha.fns_attn.dist_scale 

            if 'v2_' not in manifold:
                attn_score = dist_to_score(g_dist, alpha, bdwth, d_intrinsic=d_intrinsic)
            else:
                attn_score = dist_to_score(g_dist, alpha, bdwth, d_intrinsic=1)

            # determine furthest distance and coordinates in embedding space
            # max_g_dist = g_dist.max().item()
            # m = g_dist.view(1, -1).argmax(1)
            # max_g_dist_indices = torch.cat(((m // X_len).view(-1, 1), (m % X_len).view(-1, 1)), dim=1)[0]          

            if a > 0:            
                N_R = attn_score.sum(-1)  # row sum
                N_C = attn_score.sum(-2)  # col sum                
                K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)       
            else:
                K_tilde = attn_score

            K_tilde = K_tilde[0,0]
            D_tilde = K_tilde.sum(-1)

            # attention_weight = F.normalize(K_tilde,p=1,dim=-1)  # can do this as the attn weights are always positive

            # ----- compute spectrum of MC -----
            ##### Q = K case #####
            if config['qk_share']:
                K_hat = torch.diag(D_tilde**(-0.5)) @ K_tilde @ torch.diag(D_tilde**(-0.5))
                # check if symmetrical
                if (K_hat==K_hat.T).all():
                    K_hat_sym = K_hat
                else:
                    K_hat_sym = 0.5*(K_hat + K_hat.T)
                eigvals, eigvecs = torch.linalg.eigh(K_hat_sym)    
                eigvecs = torch.diag(D_tilde**(-0.5)) @ eigvecs

                if 'cuda' in device:                        
                    eigvals = eigvals.cpu().detach().numpy()
                    eigvecs = eigvecs.cpu().detach().numpy() 
                else:
                    eigvals = eigvals.detach().numpy()
                    eigvecs = eigvecs.detach().numpy()      
        
            ##### Q != K case #####
            else:
                # frequency (need to tune later)
                #q = 1/3
                q = 1/4*((attn_score[0,0] - attn_score[0,0].T).abs().max().item())  # q
                # Magnetic transform
                if 'cuda' in device:  
                    K_S = (attn_score[0,0] + attn_score[0,0].T).cpu().detach()/2  # symmetrized
                    K_A = -(attn_score[0,0] - attn_score[0,0].T).cpu().detach()   # anti-symmetrized                        
                else:
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

            # Shortest path length vs sequence length
            target_lengths = np.arange(50,501,50)
            shortest_path_lengths = np.zeros((len(target_lengths), int(500*500)))
            for tlen_idx in tqdm(range(len(target_lengths))):
                tlen = target_lengths[tlen_idx]
                tlen_idxs = np.where(np.array(X_lens) == tlen)[0]
                # X, Y = train_loader.dataset[target_len_idxs[idx]]
                X, Y = train_loader.dataset[tlen_idxs[idx]]
                outputs, attention_weights, g_dists = model(X[None].to(device)) 
                if 'cuda' not in device:
                    attention_weights = np.squeeze(attention_weights[0].detach().numpy())[:tlen, :tlen]
                else:
                    attention_weights = np.squeeze(attention_weights[0].cpu().detach().numpy())[:tlen, :tlen]
                edge_weights = 1/np.abs(attention_weights)
                np.fill_diagonal(edge_weights, 0)
                G = nx.from_numpy_array(edge_weights, create_using=nx.DiGraph)
                shortest_path_length = np.full(attention_weights.shape, np.inf)
                np.fill_diagonal(shortest_path_length, 0)
                for source in range(shortest_path_length.shape[0]):
                    lengths, shortest_paths = nx.single_source_dijkstra(G, source) # Compute shortest paths using Dijkstra's algorithm
                    for target in lengths:
                        shortest_path_length[source, target] = len(shortest_paths[target]) - 1
                shortest_path_lengths[tlen_idx,:int(tlen*tlen)] = shortest_path_length.flatten()
                
            # qk_affix = 'qqv' if config['qk_share'] else 'qkv'
            # Save numerical results
            alpha = attn_setup['alpha']
            np.savez(njoin(model_dir, f'attn_graph_results.npz'), 
                        bdwth=bdwth,
                        X_len=X_len,
                        eigvals=eigvals,
                        diffusion_map=eigvecs, 
                        close_xy_idxs=close_xy_idxs,
                        far_xy_idxs=far_xy_idxs, 
                        # attention_weights=attention_weight.detach() if 'cuda' not in device else attention_weight.cpu().detach(), 
                        attention_weights=attention_weights,
                        embeddings=data[0], shortest_path_lengths=shortest_path_lengths)
            # Save corresponding text
            with open(njoin(model_dir, f'text.txt'), 'w') as f:
                for word in txts:
                    f.write(f"{word}\n")                
        
    # ----- RESULTS FOR DPFORMER -----
    elif config['model_name'][-8:] == 'dpformer':

        print('Running dpformer. \n')

        attn_setup, config, run_performance, train_setting =\
            load_model_files(model_dir)

        config['dataset_name'] = attn_setup['dataset_name']
        if config['dataset_name'] == 'imdb' and 'num_classes' not in config.keys():
            config['num_classes'] = 2
        config['max_len'] = config['seq_len']
        # correction for model_name
        if 'model_name' not in config.keys():
            config['model_name'] = attn_setup['model_name'][2:] if config['is_op'] else attn_setup['model_name']            

        model = Transformer(config, is_return_dist=True)  
        model = model.to(device)   
        checkpoint = njoin(model_dir, 'ckpt.pt')
        ckpt = torch.load(checkpoint, map_location=torch.device(device))
        model.load_state_dict(ckpt['model'])
        model.eval()    
        
        # Shortest path length vs sequence length
        target_lengths = np.arange(50,501,50)
        shortest_path_lengths = np.zeros((len(target_lengths), int(500*500)))
        for tlen_idx in tqdm(range(len(target_lengths))):
            tlen = target_lengths[tlen_idx]
            tlen_idxs = np.where(np.array(X_lens) == tlen)[0]
            # X, Y = train_loader.dataset[target_len_idxs[idx]]
            X, Y = train_loader.dataset[tlen_idxs[idx]]
            outputs, attention_weights, g_dists = model(X[None].to(device))   
            if 'cuda' not in device:
                attention_weights = np.squeeze(attention_weights[0].detach().numpy())[:tlen, :tlen]
            else:
                attention_weights = np.squeeze(attention_weights[0].cpu().detach().numpy())[:tlen, :tlen]
            edge_weights = 1/np.abs(attention_weights)
            np.fill_diagonal(edge_weights, 0)
            G = nx.from_numpy_array(edge_weights, create_using=nx.DiGraph)
            shortest_path_length = np.full(attention_weights.shape, np.inf)
            np.fill_diagonal(shortest_path_length, 0)
            for source in range(shortest_path_length.shape[0]):
                lengths, shortest_paths = nx.single_source_dijkstra(G, source) # Compute shortest paths using Dijkstra's algorithm
                for target in lengths:
                    shortest_path_length[source, target] = len(shortest_paths[target]) - 1
            shortest_path_lengths[tlen_idx,:int(tlen*tlen)] = shortest_path_length.flatten()
            
        # Save numerical results
        np.savez(njoin(model_dir, f'attn_graph_results.npz'), X_len=X_len,
                attention_weights=attention_weights, 
                shortest_path_lengths=shortest_path_lengths)
        # Save corresponding text
        with open(njoin(model_dir, f'text.txt'), 'w') as f:
            for word in txts:
                f.write(f"{word}\n")     