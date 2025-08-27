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

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from UTILS.data_utils import create_examples, glove_create_examples

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/dynamic_inference.py arguments')    
    parser.add_argument('--model_dir', default='', help='Pretrained models root')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--is_dist_based', type=str2bool, nargs='?', const=True, default=False)    

    args = parser.parse_args()   

    model_dir = args.model_dir
    batch_size = args.batch_size    
    loss_fn = nn.CrossEntropyLoss()  # only loss func used in main.py    

    # Get model setting from dir
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    print(f'device: {device} \n')

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
    print('Begin dynamic inference! \n')                                              

    # ----- load pretrained_model -----    
    attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
    fix_embed = attn_setup['fix_embed']
    pretrained_model_name = config['pretrained_model_name'] if fix_embed else False
    dataset_name = config['dataset_name'] = attn_setup['dataset_name']
    if dataset_name == 'imdb' and 'num_classes' not in config.keys():
        config['num_classes'] = 2                
    config['max_len'] = config['seq_len']    
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

    # ----- load dataset -----
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
            from UTILS.data_utils import create_examples
            tokenizer = PretrainedTokenizer(pretrained_model='wiki.model', vocab_file='wiki.vocab')

            train_dataset = create_examples(main_args, tokenizer, mode='train')
            test_dataset = create_examples(main_args, tokenizer, mode='test')            

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)                           
        test_n_batches, test_n_samples = len(test_loader), len(test_loader.dataset)    

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

    print(f'Dataset {dataset_name} loaded! \n')
    # ----------------------- 

    # ----- dynamic inference -----
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