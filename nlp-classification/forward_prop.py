import argparse
import numpy as np
import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from time import time, sleep
from typing import Union
from tqdm import tqdm
from constants import DROOT, MODEL_NAMES
from mutils import njoin, create_model_dir, convert_train_history, structural_model_root
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

# quick run (single unit)
"""
python -i forward_prop.py --n_layers=1 --n_attn_heads=2 --model_name=dpformer\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode

python -i forward_prop.py --n_layers=1 --n_attn_heads=2 --model_name=rdfnsformer --alpha=1.5 --a=0.5\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode

python -i forward_prop.py --n_layers=1 --n_attn_heads=2 --model_name=sinkformer --n_it=1\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode  
"""

print('---------- Investigation of forward prop ----------')
if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='main_seq_classification.py training arguments')    
    parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--use_custom_optim', default=False, type=bool, help='to use custom optimizer')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--train_bs', default=2, type=int)
    parser.add_argument('--eval_bs', default=10, type=int)
    parser.add_argument('--epochs', default=1, type=float)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--eval_strat', default='steps', type=str)
    parser.add_argument('--eval_steps', default=200, type=int)
    parser.add_argument('--log_strat', default='steps', type=str)
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--save_steps', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int or type(None))
    parser.add_argument('--grad_accum_step', default=8, type=int)
    parser.add_argument('--debug', default=False, type=bool)  # for debuggin
    parser.add_argument('--lr_scheduler_type', default=None, type=str or type(None))
    parser.add_argument('--do_train', default=True, type=bool)
    parser.add_argument('--do_eval', default=True, type=bool)
    parser.add_argument('--wandb_log', default=False, type=bool)

    parser.add_argument('--milestones', default='', type=str or list) # Epoch units
    parser.add_argument('--gamma', default=None, type=float or type(None)) # Decay factor

    # Model settings    
    # General
    #parser.add_argument('--sparsify_type', default=None, type=str)
    parser.add_argument('--qk_share', default=False, type=bool)
    parser.add_argument('--qkv_bias', default=False, type=bool)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_attn_heads', default=2, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)    
    parser.add_argument('--model_name', default='fnsformer', type=str, help='v3fnsformer | sinkformer | dpformer') 
    # FNSformer
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--bandwidth', default=1, type=float) 
    parser.add_argument('--a', default=1, type=float, help='0 | 0.5 | 1')
    # Sinkformer      
    parser.add_argument('--n_it', default=1, type=int)

    # Dataset settings
    parser.add_argument('--max_len', default=1024, type=int)
    parser.add_argument('--dataset_name', default='imdb', type=str)
    parser.add_argument('--divider', default=1, type=int)  # downsizing the test dataset
    # Path settings
    parser.add_argument('--model_root', default='', type=str, help='root dir of storing the model')

    args = parser.parse_args()    

    if not args.wandb_log:
        os.environ["WANDB_DISABLED"] = "true"

    repo_dir = os.getcwd()  # main dir 
    dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                       if torch.cuda.is_available() else "cpu")   
    device_name = "GPU" if dev.type != "cpu" else "CPU"
    ddp = torch.distributed.is_available() and args.train_with_ddp
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

    args.model_name = args.model_name.lower()
    #assert args.model_name in MODEL_NAMES, f'{args.model_name} does not exist in {MODEL_NAMES}'

    logging.set_verbosity_debug()
    logger = logging.get_logger()

    # ---------------------------------------- 1. Dataset setup ----------------------------------------

    max_length = args.max_len    
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    def preprocess_logits_for_metrics(logits, labels):
        preds = logits.argmax(dim=-1)
        return preds
        
    # [None, 'micro', 'macro', 'weighted']
    average_type = 'micro'
    metric_acc = load_metric("accuracy")  # average=average_type
    # metric_f1 = load_metric("f1")
    # metric_prcn = load_metric("precision") 
    # metric_recall = load_metric("recall")                 
         
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        acc = metric_acc.compute(predictions=preds, references=labels)        
        #f1_score = metric_f1.compute(predictions=preds, references=labels)  
        #precision = metric_prcn.compute(predictions=preds, references=labels)        
        #recall = metric_recall.compute(predictions=preds, references=labels)                
        #return {"accuracy":acc,"f1_score":f1_score,"precision":precision,"recall":recall}
        #return {"accuracy":acc,"f1_score":f1_score}
        return {"accuracy":acc}
    
    tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                 vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                 merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                 max_length     = max_length)
    ########## add other options here ##########

    # save tokenized dataset
    tokenized_dataset_dir = njoin(DROOT, "DATASETS", f"tokenized_{args.dataset_name}")
    if not isdir(tokenized_dataset_dir):         
        print("Downloading data!") if master_process else None
        # create cache for dataset
        dataset = get_dataset(args.dataset_name, njoin(DROOT, "DATASETS"))

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
    if args.dataset_name != 'imdb':
        tokenized_dataset = process_dataset_cols(tokenized_dataset)

    keys = list(tokenized_dataset.keys())
    if len(keys) == 1:
        tokenized_dataset = tokenized_dataset[keys[0]].train_test_split(0.5)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]
    #del tokenized_dataset  # alleviate memory         

    # convert to torch.Tensor from list
    train_dataset.set_format('torch')
    eval_dataset.set_format('torch')

    # ---------- REMOVE LATER ----------  
    divider = args.divider  # downsize dataset for 
    if divider > 1:
        train_dataset = train_dataset.filter(lambda example, idx: idx % divider == 0, with_indices=True)
        eval_dataset = eval_dataset.filter(lambda example, idx: idx % divider == 0, with_indices=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #config = ModelConfig.from_json_file(f"{repo_dir}/models/config.json")    

    # ---------------------------------------- 2. Model setup ----------------------------------------

    #n_attn_headss = [1,2,4,6,8,12,24]
    n_attn_headss = [2]
    min_avg_dists = []
    max_avg_dists = []
    avg_dists = []
    head_dims = []
    max_kernel_inputs = []

    hidden_size = args.hidden_size
    for n_attn_heads in n_attn_headss:

        # paths
        if args.model_root == '':
            model_root = structural_model_root(qk_share=args.qk_share, n_layers=args.n_layers,
                                            n_attn_heads=n_attn_heads, hidden_size=hidden_size,
                                            lr=args.lr, bs=args.train_bs, 
                                            use_custom_optim=args.use_custom_optim,
                                            milestones=args.milestones, gamma=args.gamma,
                                            epochs=args.epochs                                               
                                            )       
            model_root = njoin(DROOT, model_root)


        else:
            model_root = args.model_root   
        #if not isdir(model_root): makedirs(model_root)

        config = ModelConfig.from_json_file(f"{repo_dir}/models/config_simple.json")
        config.num_labels = len(set(train_dataset['label']))   
        config.qk_share = args.qk_share
        config.qkv_bias = args.qkv_bias

        attn_setup = {'qk_share': args.qk_share, 'qkv_bias': args.qkv_bias}
        attn_setup['model_name'] = args.model_name
        attn_setup['dataset_name'] = args.dataset_name
        if 'fnsformer' in args.model_name:
            attn_setup['alpha'] = args.alpha      
            attn_setup['bandwidth'] = args.bandwidth          

            if args.model_name in ['spfnsformer', 'spopfnsformer']:

                if args.alpha < 2:
                    config.d_intrinsic = hidden_size//n_attn_heads - 1  # head_dim
                    config.sphere_radius = ((np.pi**(1/config.d_intrinsic)-1)/np.pi)   
                    #config.sphere_radius = 1
                    attn_setup['d_intrinsic'] = config.d_intrinsic
                elif args.alpha >= 2:
                    config.sphere_radius = 1

                # degree index
                attn_setup['a'] = args.a      
                config.a = args.a  
            
                # mask for distance
                config.mask_val = config.sphere_radius * np.pi
                attn_setup['sphere_radius'] = config.sphere_radius       
                attn_setup['mask_val'] = np.pi * config.sphere_radius   

            elif args.model_name in ['rdfnsformer', 'rdopfnsformer']:

                if args.alpha < 2:
                    config.d_intrinsic = args.hidden_size//n_attn_heads  # head_dim
                    config.sphere_radius = ((np.pi**(1/config.d_intrinsic)-1)/np.pi)   
                    #config.sphere_radius = 1
                    attn_setup['d_intrinsic'] = config.d_intrinsic
                elif args.alpha >= 2:
                    config.sphere_radius = 1

                # degree index
                attn_setup['a'] = args.a      
                config.a = args.a          

        elif args.model_name == 'sinkformer':
            n_it = args.n_it
            attn_setup['n_it'] = n_it
            attn_setup['bandwidth'] = args.bandwidth
            # mask for DP
            config.mask_val = -1e-9
            attn_setup['mask_val'] = config.mask_val

        config.num_hidden_layers = args.n_layers
        config.num_attention_heads = n_attn_heads
        config.hidden_size = hidden_size
        config.attention_window = args.max_len  # full attn, no sliding windows                    

        model = FNSFormerForSequenceClassification(config, **attn_setup).to(dev)    
        ########## add other model options here ##########

        models_dir, model_dir = create_model_dir(model_root, **attn_setup)
        if master_process:
            if not os.path.isdir(models_dir): os.makedirs(models_dir)
            if not os.path.isdir(model_dir): os.makedirs(model_dir)                
        
        if args.lr_scheduler_type in ['linear', 'cosine']:
            if args.warmup_steps is None:
                warmup_steps = 100   
            else:
                warmup_steps = args.warmup_steps
        else:
            warmup_steps = args.warmup_steps

        training_args_dict = {"output_dir": model_dir,                         
                            "per_device_train_batch_size": args.train_bs,
                            "per_device_eval_batch_size": args.eval_bs,
                            "num_train_epochs": args.epochs,                                                    
                            "evaluation_strategy": args.eval_strat,
                            "eval_steps": args.eval_steps,
                            "logging_strategy": args.log_strat,
                            "logging_steps": args.logging_steps,
                            "save_steps": args.save_steps,                          
                            "seed": args.seed,
                            "warmup_steps": warmup_steps,
                            "gradient_accumulation_steps": args.grad_accum_step,
                            "do_train": args.do_train,                          
                            "do_eval": args.do_eval
                            }

        if args.max_steps is not None:
            training_args_dict["max_steps"] = args.max_steps
        if args.debug is True:
            training_args_dict["debug"] = "underflow_overflow"        

        if args.use_custom_optim is False:

            if args.lr_scheduler_type is not None:            
                training_args_dict["lr_scheduler_type"] = args.lr_scheduler_type
                #training_args_dict["lr_scheduler_kwargs"] = lr_scheduler_kwargs

            training_args_dict["learning_rate"] = args.lr
            training_args_dict["weight_decay"] = args.weight_decay    
            training_args = TrainingArguments(**training_args_dict)

        else:

            # CLEAN UP LATER        

            # Create adamw_torch optimizer manually (https://github.com/huggingface/transformers/issues/18635)
            # decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # optimizer_grouped_parameters = [
            #     {
            #         "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            #         #"weight_decay": training_args.weight_decay,
            #         "weight_decay": args.weight_decay,
            #     },
            #     {
            #         "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            #         "weight_decay": 0.0,
            #     },
            # ]

            if isinstance(args.milestones,str):
                if args.milestones == '':
                    args.milestones = [int(args.epochs/2), args.epochs]
                else:
                    args.milestones = [int(str_epoch) for str_epoch in args.milestones.split(',')]             
            

        training_args = TrainingArguments(**training_args_dict)
        steps_per_train_epoch = int(len(train_dataset)/(training_args.per_device_train_batch_size*device_total*training_args.gradient_accumulation_steps ))    
        if master_process:
            print("-"*25)
            print(training_args_dict)
            if args.use_custom_optim is True:
                print(f'milestones: {args.milestones}')
                print(f'gamma: {args.gamma}')
            print('\n')
            print(f'model: {args.model_name}')
            if 'fnsformer' in args.model_name:
                print(f'alpha = {args.alpha}, bandwidth = {args.bandwidth}, a = {args.a}')
            print(f'dataset: {args.dataset_name}')
            print(attn_setup)           
            print(f'Model will be saved in {model_dir}')        
            print("-"*25 + "\n")      
                
            print("-"*25)
            print(f"steps_per_train_epoch {steps_per_train_epoch}")
            print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
            print(f"{device_name} count: {device_total}")
            print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
            print("-"*25 + "\n")

        if training_args.num_train_epochs >= 1 and args.max_steps == None:
            training_args.eval_steps    = int(steps_per_train_epoch)        
            #training_args.logging_steps = int(steps_per_train_epoch/3)  # int(steps_per_train_epoch/5)
            training_args.logging_steps = int(steps_per_train_epoch)        
            #training_args.save_steps    = int(steps_per_train_epoch)
            training_args.save_steps    = int(steps_per_train_epoch * args.epochs)
            
        trainer_kwargs = {'model': model,                      
                        'args': training_args,
                        'train_dataset': train_dataset,
                        'eval_dataset': eval_dataset,
                        'tokenizer': tokenizer,
                        'data_collator': data_collator,
                        'compute_metrics': compute_metrics,
                        'preprocess_logits_for_metrics': preprocess_logits_for_metrics                                            
                        }
        if args.use_custom_optim is True:
            from CustomTrainer import CustomTrainer

            # Create adamw_torch optimizer manually (https://github.com/huggingface/transformers/issues/18635)
            decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ] 
                    
            #params=model.parameters()
            optimizer = AdamW(optimizer_grouped_parameters, 
                            lr=args.lr,
                            betas=(training_args.adam_beta1, training_args.adam_beta2),
                            eps=training_args.adam_epsilon
                            )
            scheduler = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)     
            trainer_kwargs['optimizers'] = (optimizer, scheduler)     
            trainer = CustomTrainer(**trainer_kwargs)

            if master_process:
                print("-"*25)
                print('CustomTrainer initialized!')
                print("-"*25 + "\n")

        else:
            from transformers import Trainer    
            trainer = Trainer(**trainer_kwargs)
            # trainer_kwargs['config'] = config
            # trainer = MyTrainer(**trainer_kwargs)    
            
            if master_process:
                print("-"*25)
                print('HF Trainer initialized!')
                print("-"*25 + "\n")             
        
        import math
        from torch.nn import functional as F    

        min_Dists = []
        max_Dists = []
        avg_Dists = []

        N_batch = 25
        for batch_idx in range(N_batch):

            X = train_dataset['input_ids'][batch_idx * args.train_bs : (batch_idx+1) * args.train_bs]
            hidden_states = model.transformer.embeddings(X)

            alpha = attn_setup['alpha']
            bandwidth = attn_setup['bandwidth']
            a = attn_setup['a']

            if alpha < 2:
                d_intrinsic = attn_setup['d_intrinsic']

            # (N,B,HD)
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            query_vectors = model.transformer.encoder.layer[0].attention.self.query(hidden_states)
            value_vectors = model.transformer.encoder.layer[0].attention.self.value(hidden_states)

            #seq_len, batch_size, embed_dim = hidden_states.size()
            batch_size, seq_len, embed_dim = hidden_states.size()
            num_heads, head_dim = model.transformer.encoder.layer[0].attention.self.num_heads, model.transformer.encoder.layer[0].attention.self.head_dim                
            assert (
                embed_dim == model.transformer.encoder.layer[0].attention.self.embed_dim
            ), f"hidden_states should have embed_dim = {model.transformer.encoder.layer[0].attention.self.embed_dim}, but has {embed_dim}"


            # begin{Lipschitz-MHA}
            # if not config.qk_share:
            #     XA = query_vectors @ model.transformer.encoder.layer[0].attention.self.key.weight
            # else:
            #     XA = query_vectors @ model.transformer.encoder.layer[0].attention.self.query.weight
            # end{Lipschitz-MHA}

            # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
            if alpha < 2:
                query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            else:
                query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)
            #query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)        
            value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

            # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
            # if not model.transformer.encoder.layer[0].attention.self.qk_share:
            if not config.qk_share:
                key_vectors = model.transformer.encoder.layer[0].attention.self.key(hidden_states)
                if alpha < 2:
                    key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                else:      
                    key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B,H,N,D)              
            # directly get Euclidean dist
            #g_dist = torch.cdist(query_vectors, key_vectors, p=2)  # (H,B,N,N)        
                            
            #print(f'Min and max distance: {g_dist.min()}, {g_dist.max()}')  
            #q = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1])
            #print(f'Distance percentiles: {torch.quantile(g_dist.flatten(), q)}')

            # type 1: key_pad_mask
            # bool_mask = (attention_mask>=0).long()
            bool_mask = train_dataset['attention_mask'][batch_idx * args.train_bs : (batch_idx+1) * args.train_bs]
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,model.transformer.encoder.layer[0].attention.self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)     

            #g_dist = g_dist.masked_fill(attention_mask_expanded==0, 1e5)

            Dist = torch.cdist(query_vectors, key_vectors, p=2)
            min_Dists.append(Dist.min().item())
            max_Dists.append(Dist.max().item())
            avg_Dists.append(Dist.mean().item())

            if alpha < 2:
                # g_dist = Dist / head_dim  # (H,B,N,N)
                # g_dist = Dist * (2**(1/head_dim) - 1) / head_dim
                g_dist = Dist * (math.pi**(1/d_intrinsic) - 1) / math.sqrt(head_dim)
                kernel_input = 1 + g_dist/bandwidth**0.5
                attn_score = kernel_input**(-d_intrinsic-alpha)
            else:
                g_dist = Dist / head_dim**0.5  # (H,B,N,N)
                kernel_input = g_dist/bandwidth**0.5
                attn_score = torch.exp(-(kernel_input)**(alpha/(alpha-1)))
            #attn_score = attn_score.masked_fill(attention_mask_expanded==0, -1e9)

            attn_score_shape = attn_score.shape
            #bound = 1e9 * seq_len
            bound = 1e5    
            if a > 0:
                
                # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
                # N_R = torch.clamp(attn_score.sum(-1), min=1/bound, max=bound)  # row sum
                # N_C = torch.clamp(attn_score.sum(-2), min=1/bound, max=bound)  # col sum
                N_R = attn_score.sum(-1)  # row sum
                N_C = attn_score.sum(-2)  # col su                
                #K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)
                K_tilde = (1/N_R**a).unsqueeze(-1) * attn_score * (1/N_C**a).unsqueeze(-2)

                attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
            else:
                attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive     

            attn_weights = F.dropout(attn_weights, p=model.transformer.encoder.layer[0].attention.self.dropout, 
                                    training=model.transformer.encoder.layer[0].attention.self.training)   

            # output 1 (conventional)
            attn_output = attn_weights @ value_vectors  

            # output 2 (Lipschitz case, with contraction?)
            # begin{Lipschitz-MHA}
            # AXW_V = model.transformer.encoder.layer[0].attention.self.value(XA).view(batch_size, seq_len, num_heads, d_intrinsic).transpose(1, 2)
            # attn_output = attn_weights @ AXW_V
            # end{Lipschitz-MHA}            

        min_avg_dists.append(np.min(min_Dists))
        max_avg_dists.append(np.max(max_Dists))
        avg_dists.append(np.mean(avg_Dists))
        head_dims.append(head_dim)
        max_kernel_inputs.append(kernel_input.detach().numpy().max())

    import matplotlib.pyplot as plt
    from constants import FIGS_DIR

    plt.plot(n_attn_headss, avg_dists, label='avg')
    plt.plot(n_attn_headss, min_avg_dists, label='min')
    plt.plot(n_attn_headss, max_avg_dists, label='max')
    #plt.plot(n_attn_headss, head_dims, label='head_dim')
    plt.plot(n_attn_headss, np.sqrt(head_dims), label='sqrt(head_dim)')
    plt.plot(n_attn_headss, max_kernel_inputs, label='max kernel inputs')
    plt.legend()

    if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
    plt.savefig(njoin(FIGS_DIR, 'nheads-vs-dist.pdf'))