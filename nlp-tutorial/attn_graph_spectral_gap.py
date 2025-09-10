import argparse
import os
from os import makedirs
from os.path import isdir
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase
import numpy as np
import torch
from torch.nn import functional as F  
from UTILS.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from constants import HYP_CMAP, HYP_CNORM, FIGS_DIR, MODEL_SUFFIX
from models.rdfnsformer import RDFNSformer
from models.dpformer import DPformer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from UTILS.data_utils import glove_create_examples
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx
import torch.nn as nn
from time import time
from UTILS.dataloader import load_dataset_and_tokenizer

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
    
def eval_model(model):
    _loss = 0
    _acc = 0
    corrects = []
    lengths = []
    for batch in test_dataset:
        inputs, labels = batch
        
        inputs = inputs.to(device)
        labels = labels.to(device)            

        if model_idx == 0: # opdpformer
            outputs, _ = model(inputs[None])
        else:
            outputs, _, _ = model(inputs[None])
        outputs = outputs[0]
        
        loss = loss_fn(outputs, labels)
        _loss += loss.item()
        prediction = outputs.argmax(dim=-1)
        correct = prediction == labels
        acc = correct.sum()
        _acc += acc.item()   
        corrects.append(correct)    
        lengths.append(config['max_len'] - count_trailing_zeros(inputs))
    _acc = _acc / len(test_loader)
    _loss = _loss / len(test_loader)
    return _acc , _loss, corrects, lengths
    

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/fdm.py arguments')    
    parser.add_argument('--models_root', default='', help='Pretrained models root')
    parser.add_argument('--fns_type', default='oprdfnsformer')  # 'spopfns'+MODEL_SUFFIX
    parser.add_argument('--is_include_pad_mask', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--is_3d', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_log', default=False, type=bool)

    args = parser.parse_args()   
    
    # ----- set up -----    
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    print(f'device: {device} \n')
    
    #SELECTED_ALPHAS = None
    SELECTED_ALPHAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2]
    SELECTED_EPSS = [1]
    SELECTED_DS = [8, 16, 32, 64]
    
    N_sample = 1000
    X_lens = np.zeros(N_sample)
    spectral_gap = np.zeros((len(SELECTED_ALPHAS) + 1, len(SELECTED_DS), N_sample))
        
    models_root = args.models_root
    for dim_idx, dim in enumerate(SELECTED_DS):
        sub_models_root = njoin(models_root, f'1L-hidden={dim}-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/')
        sub_models_root = sub_models_root.replace('\\','')
        dirnames = sorted([dirname for dirname in os.listdir(sub_models_root) if 'former' in dirname])  

        # select based on bandwidth    
        DCT_ALL = collect_model_dirs(sub_models_root, suffix=MODEL_SUFFIX)
        model_types = list(DCT_ALL.keys())
        dp_types = [model_type for model_type in model_types if 'dp' in model_type]
        
        
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
            
        # ----- fns setting -----
        alphas = sorted(df_model.loc[:,'alpha'].unique())  # small to large
        epss = sorted(df_model.loc[:,'bandwidth'].unique())  
        
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
            
            # Include opdpformer
            model_dirs.append(njoin(DCT_ALL['opdpformer'].loc[0,'model_dir'], f'model={seed}'))
            
            print(f'Bandwidth: {eps}, seed selected: {seed}')
            
            model_dirs = sorted(model_dirs)
        
            for model_idx, model_dir in enumerate(model_dirs):

                # ----- load pretrained_model -----    
                attn_setup, config, run_performance, train_setting = load_model_files(model_dir)

                fix_embed = attn_setup['fix_embed']
                pretrained_model_name = config['pretrained_model_name'] if fix_embed else False

                config['dataset_name'] = attn_setup['dataset_name']
                config['max_len'] = config['seq_len']
                
                is_fns = attn_setup['model_name'][-9:] == 'fns' + MODEL_SUFFIX 
                if not is_fns: # opdpformer
                    model = DPformer(config, is_return_dist=True)
                else: # rdfnsformer
                    model = RDFNSformer(config, is_return_dist=True) 
                    manifold = attn_setup['manifold']      
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
                    for ii in range(len(train_loader.dataset)):
                        X, Y = train_loader.dataset[ii]
                        X_lens.append(config['max_len'] - count_trailing_zeros(X) )
                    
                    # Random downsample
                    np.random.seed(0)
                    sample_idxs = np.random.choice(len(X_lens), N_sample, replace=False)
                    
                for i, idx in tqdm(enumerate(sample_idxs)):
                    X, Y = train_loader.dataset[idx]
                    X_len = config['max_len'] - count_trailing_zeros(X)
                    X_lens[i] = X_len
                    _, attention_weights, g_dists = model(X[None])  
                    # if is_fns:
                    #     _, attention_weights, g_dists = model(X[None])  
                    # else:
                    #     _, attention_weights = model(X[None])  
                    # Find spectral gap
                    for lidx, g_dist in enumerate(attention_weights):
                        attention_weights = attention_weights[lidx][0,0,:X_len,:X_len].detach().numpy()
                        eigvals, _ = np.linalg.eig(attention_weights)
                        sort_idx = np.argsort(eigvals)
                        eigvals = eigvals[sort_idx]
                        spectral_gap[model_idx, dim_idx, i] = eigvals[-1] - eigvals[-2]

    ##### Save #####
    if 'L-hidden' in args.models_root.split('/')[1]:
        SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1])
    else:
        SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1], 
                        args.models_root.split('/')[2])
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    

    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)               
    # np.savez(njoin(SAVE_DIR, fig_file), indices=indices, X_lens=X_lens)
    spectrum_file = f'all_spectral_gap.npz'
    np.savez(njoin(SAVE_DIR, spectrum_file), spectral_gap=spectral_gap, X_lens=X_lens, alphas=alphas, dimensions=SELECTED_DS)
                    
                
                # # Evaluate each model
                # acc , loss, correct, lengths = eval_model(model)
                # np.savez(njoin(SAVE_DIR, eval_file), acc=acc, loss=loss, correct=correct, lengths=lengths)
                
            #     ##### Plot #####
            #     # index_names = ['node_connectivity', 'edge_connectivity', 'algebraic_connectivity',
            #     #            'average_shortest_path_length', 'diameter', 'assortativity_coefficient',
            #     #            'rich_club_coefficient']
            #     index_names = ['average_shortest_path_length', 'diameter', 'assortativity_coefficient', 'spectral gap']
            #     for index_idx in range(5):
            #         try:
            #             ax = axs[index_idx,alphas.index(alpha)]
            #             ax.hist(indices[:,index_idx], bins=100)
            #             ax.set_title(f'alpha={alpha:.2f}, bandwidth={bandwidth:.2f}')
            #             ax.set_ylabel("Probability density")
            #             ax.set_xlabel(index_names[index_idx])
            #         except:
            #             continue
                    
            # if 'L-hidden' in args.models_root.split('/')[1]:
            #     SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1])
            # else:
            #     SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1], 
            #                     args.models_root.split('/')[2])
            # if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    

            # if not isdir(SAVE_DIR): makedirs(SAVE_DIR)   
            # fig_file = f'attn_graph-{args.fns_type}-eps={eps}-pretrained_embd={pretrained_model_name}-same_token={args.use_same_token}.pdf'
            # plt.tight_layout()
            # plt.savefig(njoin(SAVE_DIR, fig_file))            
            # print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')    