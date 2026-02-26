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
from UTILS.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files
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
        # rad = 5.95
        rad = 5
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
                 
            # --------------------------------------------------------------------------------

            """
            if X_len < config['max_len'] and not args.is_include_pad_mask:
                g_dist = g_dist[:,:,:X_len,:X_len]
            else:
                g_dist = g_dist[:,:,:,:]
                X_len = config['max_len']      

            # determine furthest distance and coordinates in embedding space
            max_g_dist = g_dist.max().item()
            m = g_dist.view(1, -1).argmax(1)
            max_g_dist_indices = torch.cat(((m // X_len).view(-1, 1), (m % X_len).view(-1, 1)), dim=1)[0]          
            """
            
            # Shortest path length vs sequence length
            # target_lengths = np.arange(50,501,50)
            target_lengths = [500]
            shortest_path_lengths = np.zeros((len(target_lengths), int(500*500)))
            for tlen_idx, tlen in enumerate(target_lengths):
                tlen_idxs = np.where(np.array(X_lens) == tlen)[0]
                # X, Y = train_loader.dataset[target_len_idxs[idx]]
                for idx in range(len(tlen_idxs)):
                    X, Y = train_loader.dataset[tlen_idxs[idx]]

                    # embedding distance
                    # ------------------------------------------------------------
                    pad_id = 0
                    positions = torch.arange(X[None].size(1), device=X[None].device, dtype=X[None].dtype).repeat(X[None].size(0), 1) + 1
                    position_pad_mask = X[None].eq(pad_id)
                    positions.masked_fill_(position_pad_mask, 0)

                    embeddings = model.embedding(X[None].to(device)).to(device) + model.pos_embedding(positions.to(device)).to(device)

                    if 'cuda' in device:
                        data = embeddings.cpu().detach().numpy() # No normalise
                    else:
                        data = embeddings.detach().numpy() # No normalise                                                                                        

                    # truncate paddings
                    data = data[:,:tlen,:]

                    center = data[0].mean(0)
                    far_xy_idxs, close_xy_idxs = [], []
                    for ii, embd_coordinates in enumerate(data[0]):
                        if ((embd_coordinates - center)**2).sum() >= rad**2:
                            far_xy_idxs.append(ii)
                        else:
                            close_xy_idxs.append(ii)
                    close_xy_idxs = np.array(close_xy_idxs)
                    far_xy_idxs = np.array(far_xy_idxs)   
                    far_xy_idxs = far_xy_idxs[far_xy_idxs < tlen]
                    # ------------------------------------------------------------

                    txts = tokenizer.convert_ids_to_tokens(list(X[None][0].detach().numpy()))
                    # print(txts)
                    print(f'tlen = {tlen}')
                    print(np.array(txts)[far_xy_idxs])
                    print(''.join(txts[:tlen]).replace(txts[0][0],' '))
                    print('\n')

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
            """
            np.savez(njoin(model_dir, f'attn_graph_results.npz'), 
                        bdwth=bdwth,
                        X_len=X_len,
                        eigvals=eigvals,
                        diffusion_map=eigvecs, 
                        close_xy_idxs=close_xy_idxs,
                        far_xy_idxs=far_xy_idxs, 
                        attention_weights=attention_weight.detach() if 'cuda' not in device else attention_weight.cpu().detach(), 
                        embeddings=data[0], shortest_path_lengths=shortest_path_lengths)
            # Save corresponding text
            with open(njoin(model_dir, f'text.txt'), 'w') as f:
                for word in txts:
                    f.write(f"{word}\n")   
            """
        
    # ----- RESULTS FOR DPFORMER -----
    elif config['model_name'][-8:] == 'dpformer':

        print('Running dpformer. \n')

        attn_setup_other, config_other, run_performance_other, train_setting_other =\
            load_model_files(model_dir)

        config_other['dataset_name'] = attn_setup_other['dataset_name']
        if config_other['dataset_name'] == 'imdb' and 'num_classes' not in config_other.keys():
            config_other['num_classes'] = 2
        config_other['max_len'] = config_other['seq_len']
        # correction for model_name
        if 'model_name' not in config_other.keys():
            config_other['model_name'] = attn_setup_other['model_name'][2:] if config_other['is_op'] else attn_setup_other['model_name']            

        model_other = Transformer(config_other, is_return_dist=True)  
        model_other = model_other.to(device)   
        checkpoint_other = njoin(model_dir, 'ckpt.pt')
        ckpt_other = torch.load(checkpoint_other, map_location=torch.device(device))
        model_other.load_state_dict(ckpt_other['model'])
        model_other.eval()    
        outputs_other, attention_weights_other, g_dists_other = model_other(X[None].to(device))   
        g_dist_other = g_dists_other[0]              
        attention_weight_other = attention_weights_other[0]  
        
        # Shortest path length vs sequence length
        target_lengths = np.arange(50,501,50)
        shortest_path_lengths = np.zeros((len(target_lengths), int(500*500)))
        for tlen_idx, tlen in enumerate(target_lengths):
            tlen_idxs = np.where(np.array(X_lens) == tlen)[0]
            X, Y = train_loader.dataset[target_len_idxs[idx]]
            outputs, attention_weights, g_dists = model_other(X[None].to(device))   
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
        """
        np.savez(njoin(model_dir, f'attn_graph_results.npz'), X_len=X_len,
                attention_weights=attention_weight_other.detach() if 'cuda' not in device else attention_weight_other.cpu().detach(), 
                shortest_path_lengths=shortest_path_lengths)
        # Save corresponding text
        with open(njoin(model_dir, f'text.txt'), 'w') as f:
            for word in txts:
                f.write(f"{word}\n")   
        """