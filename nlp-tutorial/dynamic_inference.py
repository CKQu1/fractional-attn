import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from ast import literal_eval
from torch.nn import functional as F  
from tqdm import tqdm
from os import makedirs
from os.path import isdir

from models.rdfnsformer import RDFNSformer
from models.dpformer import DPformer

from constants import *
from utils.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from utils.figure_utils import matrixify_axs, label_axs

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from utils.data_utils import create_examples, glove_create_examples

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    """
    python -i dynamic_inference.py\
     --models_root=.droot/1L-hidden\=16-max_len\=None-glove/config_qqv/imdb/layers\=1-heads\=1-qqv/
    """

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/dynamic_inference.py arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--models_root', default='', help='Pretrained models root')
    parser.add_argument('--fns_type', default='oprdfnsformer')  # 'spopfns'+MODEL_SUFFIX
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--is_dist_based', type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()   

    batch_size = args.batch_size
    loss_fn = nn.CrossEntropyLoss()

    # ---------- set up ----------

    # Get model setting from dir
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    models_root = args.models_root.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(models_root) if 'former' in dirname])  

    # select based on bandwidth    
    DCT_ALL = collect_model_dirs(models_root, suffix=MODEL_SUFFIX)
    model_types = list(DCT_ALL.keys())
    dp_types = [model_type for model_type in model_types if 'dp' in model_type]
    is_dp_trained = len(dp_types) > 0

    #SELECTED_ALPHAS = None
    SELECTED_ALPHAS = [1.0, 2.0]
    #EXCLUDED_EPSS = None
    EXCLUDED_EPSS = [100, 10, 0.1, 0.01, 0.001]
    for model_type in model_types:
        if args.fns_type in model_type:            
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            if SELECTED_ALPHAS is not None:
                # ----- filter alphas -----
                df_model = df_model[df_model['alpha'].isin(SELECTED_ALPHAS)]
                # ------------------------
                df_model.reset_index(drop=True, inplace=True)
            if EXCLUDED_EPSS is not None:
                # ----- filter alphas -----
                df_model = df_model[~df_model['bandwidth'].isin(EXCLUDED_EPSS)]
                # ------------------------
                df_model.reset_index(drop=True, inplace=True)
            break

    # ----- general settings -----
    # num_attention_heads, num_hidden_layers, hidden_size = df_modedm_l.loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())  # small to large
    epss = sorted(df_model.loc[:,'bandwidth'].unique())      

    nrows, ncols = 1, 2
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'
    axs = matrixify_axs(axs, nrows, ncols)
    label_axs(fig, axs)

    if args.is_dist_based:
        #controlled_variables = [2**(-p) for p in range(-4,3)]
        #controlled_variables = [10**(-p) for p in range(-2,4)]
        controlled_variables = [10**(-p) for p in [-2,0,15]]
        #controlled_variables = [0]
    else:
        #controlled_variables = np.arange(0,1.2,0.2)
        controlled_variables = np.arange(0,1.1,0.1)
    print('Begin dynamic inference! \n')
    
    for eps_idx, eps in enumerate(epss):        

        # collect DP type
        is_eval_dp = eps_idx == 0 and len(dp_types) > 0
        if is_eval_dp:
            df_dp = DCT_ALL[dp_types[0]]                        
            if df_dp.loc[0,'ensembles'] > 0:
                dp_seeds = literal_eval(str(sorted(df_model.loc[0,'seeds']))) 
            dp_metric_dynamic = np.zeros([1, 2, len(controlled_variables), len(dp_seeds)])
                
        for alpha_idx, alpha in enumerate(alphas):
            # collect FNS type
            seeds = literal_eval(str(sorted(df_model.loc[alpha_idx,'seeds'])))      
            if alpha_idx == 0:
                metric_dynamic = np.zeros([df_model.shape[0], 2, len(controlled_variables), len(seeds)])      
            for seed_idx, seed in tqdm(enumerate(seeds)):
                torch.manual_seed(seed)
                fns_dir = njoin(df_model.loc[alpha_idx,'model_dir'], f'model={seed}')

                # ----- load pretrained_model -----    
                attn_setup, config, run_performance, train_setting = load_model_files(fns_dir)
                fix_embed = attn_setup['fix_embed']
                pretrained_model_name = config['pretrained_model_name'] if fix_embed else False
                dataset_name = config['dataset_name'] = attn_setup['dataset_name']
                if dataset_name == 'imdb' and 'num_classes' not in config.keys():
                    config['num_classes'] = 2                
                config['max_len'] = config['seq_len']
                main_args = AttrDict(config)
                is_do_once = (alpha_idx==0) and (seed_idx==0)
                
                if is_do_once:
                    qk_share = config['qk_share']
                    # load dataset
                    if dataset_name.lower() == 'imdb':
                        if fix_embed:
                            if config['pretrained_model_name'] == 'glove':
                                from constants import GLOVE_DIMS
                                for glove_dim in GLOVE_DIMS:
                                    if glove_dim >= config['hidden']:
                                        break                
                                tokenizer = get_tokenizer("basic_english")
                                glove = GloVe(name='6B', dim=glove_dim)
                                vocab_size = len(glove.stoi)   
                                train_dataset = glove_create_examples(main_args, glove_dim, tokenizer, mode='train')
                                test_dataset = glove_create_examples(main_args, glove_dim, tokenizer, mode='test')  

                            elif config['pretrained_model_name'] == 'bert-base-uncased':
                                from transformers import AutoTokenizer
                                #tokenizer = AutoTokenizer.from_pretrained(f'distilbert/{args.pretrained_model_name}')
                                tokenizer = AutoTokenizer.from_pretrained(f'distilbert/distilbert-base-uncased')
                                train_dataset = create_examples(main_args, tokenizer, mode='train')
                                test_dataset = create_examples(main_args, tokenizer, mode='test')

                            elif config['pretrained_model_name'] == 'albert-base-v2':
                                from transformers import AlbertTokenizer
                                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                                train_dataset = create_examples(main_args, tokenizer, mode='train')
                                test_dataset = create_examples(main_args, tokenizer, mode='test')
                        else:
                            from tokenization import Tokenizer, PretrainedTokenizer
                            from utils.data_utils import create_examples
                            tokenizer = PretrainedTokenizer(pretrained_model='wiki.model', vocab_file='wiki.vocab')

                            train_dataset = create_examples(main_args, tokenizer, mode='train')
                            test_dataset = create_examples(main_args, tokenizer, mode='test')            

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)                           
                        test_n_batches, test_n_samples = len(test_loader), len(test_loader.dataset)    

                    def eval_model(model):
                        dynamic_inference_acc, dynamic_inference_loss = 0, 0  
                        for batch in test_loader:
                            inputs, labels = batch
                            
                            inputs = inputs.to(device)
                            labels = labels.to(device)            

                            outputs, attention_weights = model(inputs)
                            
                            loss = loss_fn(outputs, labels)
                            dynamic_inference_loss += loss.item()
                            acc = (outputs.argmax(dim=-1) == labels).sum()
                            dynamic_inference_acc += acc.item()       
                        dynamic_inference_acc = dynamic_inference_acc / test_n_samples
                        dynamic_inference_loss = dynamic_inference_loss / test_n_batches
                        return dynamic_inference_acc , dynamic_inference_loss

                    print(f'Dataset {dataset_name} loaded! \n')

                # ----- fdm -----
                alpha = alphas[alpha_idx]              
                d_intrinsic = attn_setup['d_intrinsic'] if alpha < 2 else None            

                for var_ii, controlled_variable in enumerate(controlled_variables):
                    ##### Distance based #####
                    if args.is_dist_based:  
                        config['dist_threshold'] = controlled_variable    
                    ##### Probability based #####
                    else:  
                        config['is_add_eval_mask'] = True                      
                        eval_mask = torch.bernoulli(torch.full((config['max_len'],config['max_len']),
                                                    controlled_variable))
                        config['eval_mask'] = eval_mask.bool().to(device)
                        
                    model_dirs = [fns_dir]
                    if is_dp_trained and alpha_idx == 0 and eps_idx == 0 and seed in dp_seeds:
                        model_dirs.append(njoin(df_dp.loc[0,'model_dir'], f'model={seed}'))                    

                    for model_idx, model_dir in enumerate(model_dirs):                        
                        if model_idx == 0:      
                            model = RDFNSformer(config)                                              
                        elif model_idx == 1 and is_eval_dp:
                            dp_attn_setup, dp_config, _, _ = load_model_files(model_dir)
                            dp_config['max_len'] = dp_config['seq_len']    
                            if dataset_name == 'imdb' and 'num_classes' not in dp_config.keys():
                                dp_config['num_classes'] = 2                                                      
                            ##### Distance based #####
                            if args.is_dist_based:  
                                dp_config['dist_threshold'] = controlled_variable    
                            ##### Probability based #####
                            else:  
                                dp_config['is_add_eval_mask'] = True                      
                                eval_mask = torch.bernoulli(torch.full((dp_config['max_len'],
                                                            dp_config['max_len']),
                                                            controlled_variable))
                                dp_config['eval_mask'] = eval_mask.bool().to(device)
                            model = DPformer(dp_config)
                        model = model.to(device)
                        checkpoint = njoin(model_dir, 'ckpt.pt')
                        if 'cpu' in device:
                            ckpt = torch.load(checkpoint, map_location=torch.device(device))                            
                        else:
                            ckpt = torch.load(checkpoint)
                        model.load_state_dict(ckpt['model'])                            
                        model.eval()
                        
                        acc, loss = eval_model(model)
                        if model_idx == 0:
                            metric_dynamic[alpha_idx,:,var_ii,seed_idx] = acc, loss                    
                        elif model_idx == 1 and is_eval_dp:
                            dp_metric_dynamic[alpha_idx,:,var_ii,seed_idx] = acc, loss

    marker = DEPTH_TO_MARKER[config['n_layers']]
    for alpha_idx, alpha in enumerate(alphas):
        acc_mean = metric_dynamic[alpha_idx,0,:,:].mean(-1)
        acc_std = metric_dynamic[alpha_idx,0,:,:].std(-1)
        loss_mean = metric_dynamic[alpha_idx,1,:,:].mean(-1)
        loss_std = metric_dynamic[alpha_idx,1,:,:].std(-1)

        c_hyp = HYP_CMAP(HYP_CNORM(alpha))                            
        axs[0,0].plot(controlled_variables, acc_mean,
                      marker=marker, markersize=MARKERSIZE,
                      c=c_hyp, linestyle=LINESTYLE_DICT[args.fns_type])
        axs[0,1].plot(controlled_variables, loss_mean,
                      marker=marker, markersize=MARKERSIZE,
                      c=c_hyp, linestyle=LINESTYLE_DICT[args.fns_type])      

        # axs[0,0].fill_between(controlled_variables,  acc_mean - acc_std, acc_mean + acc_std,
        #                       color=c_hyp, alpha=1/2)                        
        # axs[0,1].fill_between(controlled_variables, loss_mean - loss_std, loss_mean + loss_std,
        #                       color=c_hyp, alpha=1/2)          
         
    if is_eval_dp:
        acc_mean = dp_metric_dynamic[0,0,:,:].mean(-1)
        acc_std = dp_metric_dynamic[0,0,:,:].std(-1)
        loss_mean = dp_metric_dynamic[0,1,:,:].mean(-1)
        loss_std = dp_metric_dynamic[0,1,:,:].std(-1)

        c_hyp = HYP_CMAP(HYP_CNORM(alpha))                            
        axs[0,0].plot(controlled_variables, acc_mean,
                    marker=marker, markersize=MARKERSIZE,
                    c=OTHER_COLORS_DICT[dp_types[0]],
                    linestyle=LINESTYLE_DICT[dp_types[0]])
        axs[0,1].plot(controlled_variables, loss_mean,
                    marker=marker, markersize=MARKERSIZE,
                    c=OTHER_COLORS_DICT[dp_types[0]],
                    linestyle=LINESTYLE_DICT[dp_types[0]])        

    # legends
    for alpha_idx, alpha in enumerate(alphas):
        c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
        axs[0,0].plot([], [], marker=marker, c=c_hyp, linestyle=LINESTYLE_DICT[args.fns_type],
                      label=rf'$\alpha$ = {alpha}')    
    if is_eval_dp:
        axs[0,0].plot([], [],
                      marker=marker, c=OTHER_COLORS_DICT[dp_types[0]],
                      linestyle=LINESTYLE_DICT[dp_types[0]])                      
                        
    #axs[0,0].invert_xaxis(); axs[0,1].invert_xaxis()    

    axs[0,0].set_title('Accuracy'); axs[0,1].set_title('Loss')
    if args.is_dist_based:
        axs[0,0].set_xlabel('Distance threshold'); axs[0,1].set_xlabel('Distance threshold')
        axs[0,0].set_xscale('log'); axs[0,1].set_xscale('log')
    else:
        axs[0,0].set_xlabel('Removal probability'); axs[0,1].set_xlabel('Removal probability')
    axs[0,0].legend(frameon=False)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label   

    dataset = main_args['dataset_name']
    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    if 'L-hidden' in args.models_root.split('/')[1]:
        SAVE_DIR = njoin(FIGS_DIR, 'nlp-task', args.models_root.split('/')[1])
    else:
        SAVE_DIR = njoin(FIGS_DIR, 'nlp-task', args.models_root.split('/')[1], 
                         args.models_root.split('/')[2])
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    
    qkv = 'qqv' if qk_share else 'qkv'
    if args.is_dist_based:           
        fig_file = f'dynamic_inference_dist-{args.fns_type}-{qkv}.pdf'
    else:
        fig_file = f'dynamic_inference_prob-{args.fns_type}-{qkv}.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file))            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')