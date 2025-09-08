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
from os.path import isdir, isfile

from models.rdfnsformer import RDFNSformer
from models.dpformer import DPformer

from constants import *
from UTILS.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from UTILS.figure_utils import matrixify_axs, label_axs
from UTILS.dataloader import load_dataset_and_tokenizer

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from UTILS.data_utils import create_examples, glove_create_examples

"""
There are 2 modes for this:
    - normal mode: args.is_normal_mode == True
        - obtain the classification errors for all single sequence length
        - batch_size is hard set to 1
    - dynamic mode: args.is_normal_mode == False
        - randomly mask attention score
        - batch_size can be anything
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/dynamic_inference.py arguments')    
    parser.add_argument('--model_dir', default='', help='Pretrained models root')
    parser.add_argument('--is_normal_mode', type=str2bool, nargs='?', const=True, default=False)
    # ----- is_normal_mode == False -----
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--is_dist_based', type=str2bool, nargs='?', const=True, default=False)    

    args = parser.parse_args()   

    model_dir = args.model_dir
    batch_size = args.batch_size if not args.is_normal_mode else 1  # for normal mode batch size hard set to 1    
    loss_fn = nn.CrossEntropyLoss()  # only loss func used in main.py    

    # Get model setting from dir
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    print(f'device: {device} \n')

    print(f'Begin dynamic inference: is_normal_mode = {args.is_normal_mode}! \n')                                              

    # ----- load pretrained_model -----    
    attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
    fix_embed = attn_setup['fix_embed']
    pretrained_model_name = config['pretrained_model_name'] if fix_embed else False
    dataset_name = config['dataset_name'] = attn_setup['dataset_name']
    if dataset_name == 'imdb' and 'num_classes' not in config.keys():
        config['num_classes'] = 2                
    max_len = config['max_len'] = config['seq_len']    
    main_args = AttrDict(config)
    
    # ---------- set up ----------    
    seed = attn_setup['seed']
    qk_share = config['qk_share']
    model_name = attn_setup['model_name']
    is_fns = model_name[-9:] == 'fns' + MODEL_SUFFIX  # model type
    checkpoint = njoin(model_dir, 'ckpt.pt')
    is_checkpoint_exist = isfile(checkpoint)

    torch.manual_seed(seed)
    if isfile(njoin(model_dir, 'run_performance.csv')) and is_checkpoint_exist:
        run = pd.read_csv(njoin(model_dir, 'run_performance.csv'))
    else:
        print(f'model={seed} incomplete training!')
        quit()            

    # ---------------------------------------- poor man's data loader ----------------------------------------
    # tokenizer_name is hard set in main.py for the following case
    if main_args.dataset_name == 'imdb' and not main_args.fix_embed:
        main_args.tokenizer_name = 'sentencepiece'  
        main_args.pretrained_model = 'wiki.model'
        main_args.vocab_file = 'wiki.vocab'
    tokenizer, train_loader, test_loader, train_size, eval_size, steps_per_epoch, num_classes =\
        load_dataset_and_tokenizer(main_args, batch_size)     
    test_n_batches, test_n_samples = len(test_loader), len(test_loader.dataset)  
    # -------------------------------------------------------------------------------------------------------- 
    print(f'Dataset {dataset_name} loaded! \n')

    # ----- varying sequence len -----
    if args.is_normal_mode:
        from UTILS.data_utils import count_trailing_zeros

        def eval_model(model):
            dynamic_metrics = []
            for dataset_loader in [train_loader, test_loader]:                  
                n_batches = len(dataset_loader)
                n_samples = len(dataset_loader.dataset)  
                dataset_metrics = np.zeros([n_samples,2])
                for batch_idx, batch in tqdm(enumerate(dataset_loader)):  # batch_size should be 1 here only
                    inputs, labels = batch
                    
                    inputs, labels = inputs.to(device), labels.to(device)            

                    outputs, attention_weights = model(inputs)
                                   
                    dataset_metrics[batch_idx,0] = int((outputs.argmax(dim=-1) == labels).item())  # double-check
                    dataset_metrics[batch_idx,1] = max_len - count_trailing_zeros(inputs[0])      
                dynamic_metrics.append(dataset_metrics)
            return dynamic_metrics

        # load weights  
        model = RDFNSformer(config) if is_fns else DPformer(config)
        model = model.to(device)                                  
        ckpt = torch.load(checkpoint, map_location=torch.device(device))                            
        model.load_state_dict(ckpt['model'])                            
        model.eval()  

        # ----- save data -----
        col_names = ['is_correct', 'seq_len']
        trainset_metrics, testset_metrics = eval_model(model)        
        df_train = pd.DataFrame(data=trainset_metrics, columns=col_names)           
        df_test = pd.DataFrame(data=testset_metrics, columns=col_names)   
        # order based on second col (seq len)
        df_train = df_train.sort_values(by='seq_len').reset_index(drop=True)
        df_test = df_test.sort_values(by='seq_len').reset_index(drop=True)
        df_train.to_csv(njoin(model_dir, f'bs=1-train_inference.csv'))
        df_test.to_csv(njoin(model_dir, f'bs=1-test_inference.csv'))        

    # ----- dynamic inference -----
    else:  

        def eval_model(model):
            dynamic_metrics = []
            for dataset_loader in [train_loader, test_loader]:
                dynamic_inference_acc, dynamic_inference_loss = 0, 0  
                n_batches = len(dataset_loader)
                n_samples = len(dataset_loader.dataset)  
                for batch in dataset_loader:
                    inputs, labels = batch
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)            

                    outputs, attention_weights = model(inputs)
                    
                    loss = loss_fn(outputs, labels)
                    dynamic_inference_loss += loss.item()
                    acc = (outputs.argmax(dim=-1) == labels).sum()
                    dynamic_inference_acc += acc.item()       
                dynamic_inference_acc = dynamic_inference_acc / n_samples
                dynamic_inference_loss = dynamic_inference_loss / n_batches
                dynamic_metrics += [dynamic_inference_acc , dynamic_inference_loss]
            return dynamic_metrics

        if args.is_dist_based:
            #controlled_variables = [2**(-p) for p in range(-4,3)]
            #controlled_variables = [10**(-p) for p in range(-2,4)]
            controlled_variables = [10**(-p) for p in [-2,0,15]]
            #controlled_variables = [0]
            controlled_variable_name = 'dist'
        else:
            #controlled_variables = np.arange(0,1.2,0.2)
            controlled_variables = np.arange(0,1.1,0.1)
            controlled_variable_name = 'prob'

        metric_dynamic = np.zeros([len(controlled_variables), 5])
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

            # load weights  
            model = RDFNSformer(config) if is_fns else DPformer(config)
            model = model.to(device)                                  
            ckpt = torch.load(checkpoint, map_location=torch.device(device))                            
            model.load_state_dict(ckpt['model'])                            
            model.eval()  

            # evaluate
            metric_dynamic[var_ii,:] = [controlled_variable] + eval_model(model)

        # ----- save data -----
        col_names = ['controlled_variable', 'train_acc', 'train_loss', 'test_acc', 'test_loss']
        df = pd.DataFrame(data=metric_dynamic, columns=col_names)   
        print(df) 
        df.to_csv(njoin(model_dir, f'{controlled_variable_name}-bs={args.batch_size}-inference.csv'))