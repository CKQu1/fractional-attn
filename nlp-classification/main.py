import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from time import time, sleep
from typing import Union
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
python -i main.py --n_layers=1 --n_attn_heads=2 --model_name=dpformer\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode

python -i main.py --n_layers=1 --n_attn_heads=2 --model_name=rdfnsformer\
 --alpha=1.5 --bandwidth=0.5 --a=1\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode

python -i main.py --n_layers=1 --n_attn_heads=2 --model_name=sinkformer --n_it=1\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode  
"""

"""
torchrun --nnodes=1 --nproc_per_node=4 main.py --n_layers=1 --n_attn_heads=2 --model_name=v3fnsformer --alpha=1.5\
 --max_len=256 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=1 --warmup_steps=0 --grad_accum_step=1 --dataset_name=rotten_tomatoes\
 --model_root=.droot/debug-mode
"""

#torch.autograd.set_detect_anomaly(True)  # delete
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
    assert args.model_name in MODEL_NAMES, f'{args.model_name} does not exist in {MODEL_NAMES}'

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

    # paths
    if args.model_root == '':
        model_root = structural_model_root(qk_share=args.qk_share, n_layers=args.n_layers,
                                           n_attn_heads=args.n_attn_heads, hidden_size=args.hidden_size,
                                           lr=args.lr, bs=args.train_bs, 
                                           use_custom_optim=args.use_custom_optim,
                                           milestones=args.milestones, gamma=args.gamma,
                                           epochs=args.epochs                                               
                                           )       
        model_root = njoin(DROOT, model_root)


    else:
        model_root = args.model_root   
    #if not isdir(model_root): makedirs(model_root)

    #config = ModelConfig.from_json_file(f"{repo_dir}/models/config_simple.json")
    config = {"attention_mode": "fnsformer",
              "attention_probs_dropout_prob": 0.1,
              #"attention_window": [64],
              "bos_token_id": 0,
              "eos_token_id": 2,
              "gradient_checkpointing": False,
              "hidden_act": "gelu",
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "ignore_attention_mask": False,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-05,
              "max_position_embeddings": 4098,
              "model_type": "fnsformer",
              "num_attention_heads": 2,
              "num_hidden_layers": 1,
              "pad_token_id": 1,
              "sep_token_id": 2,
              "type_vocab_size": 1,
              "vocab_size": 50265,
              "num_labels": len(set(train_dataset['label'])),
              "qk_share": args.qk_share,
              "qkv_bias": args.qkv_bias,
              "num_hidden_layers": args.n_layers,
              "num_attention_heads": args.n_attn_heads,
              "hidden_size": args.hidden_size,
              "attention_window": args.max_len  # full attn, no sliding windows
              }         

    attn_setup = {'qk_share': args.qk_share, 'qkv_bias': args.qkv_bias}
    attn_setup['model_name'] = args.model_name
    attn_setup['dataset_name'] = args.dataset_name
    if 'fnsformer' in args.model_name:
        attn_setup['alpha'] = args.alpha      
        attn_setup['bandwidth'] = args.bandwidth          

        if args.model_name in ['v2fnsformer', 'v3fnsformer', 'v4fnsformer', 'opfnsformer', 'v2opfnsformer']:

            if args.alpha < 2:
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = int(args.hidden_size//args.n_attn_heads - 1)  # head_dim
                config['sphere_radius'] = ((np.pi**(1/config.d_intrinsic)-1)/np.pi)   
                #config.sphere_radius = 1                
            elif args.alpha >= 2:
                config['sphere_radius'] = attn_setup['d_intrinsic'] = 1                 
        
            # mask for distance
            config['mask_val'] = attn_setup['mask_val'] = config.sphere_radius * np.pi
            attn_setup['sphere_radius'] = config.sphere_radius                      

        elif args.model_name in ['rdfnsformer', 'rdopfnsformer']:

            if args.alpha < 2:
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = int(args.hidden_size/args.n_attn_heads)  # head_dim                

        # degree index
        config['a'] = attn_setup['a'] = args.a      

    elif args.model_name == 'sinkformer':
        n_it = args.n_it
        attn_setup['n_it'] = n_it
        attn_setup['bandwidth'] = args.bandwidth
        # mask for DP
        config['mask_val'] = attn_setup['mask_val'] = -1e-9                         

    models_dir, model_dir = create_model_dir(model_root, **attn_setup)
    if master_process:
        if not os.path.isdir(models_dir): os.makedirs(models_dir)
        if not os.path.isdir(model_dir): os.makedirs(model_dir) 
          
    # save config
    with open(njoin(model_dir,"config.json"), "w") as ofile: 
        json.dump(config, ofile)   
    # save attn_setup
    with open(njoin(model_dir,"attn_setup.json"), "w") as ofile: 
        json.dump(attn_setup, ofile)        

    model_config = ModelConfig.from_json_file(njoin(model_dir, 'config.json'))
    model = FNSFormerForSequenceClassification(model_config, **attn_setup).to(dev)        
    ########## add other model options here ##########            
            
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

    # if args.model_name == 'fnsformer':
    #     trainer_kwargs['config'] = config
    #     # if args.sparsify_type == None:
    #     #     if args.with_frac:
    #     #         args.sparsify_type = 'longformer'
    #     #     else:
    #     #         args.sparsify_type = 'diffuser'
    #     #     trainer_kwargs['sparsify_type'] = args.sparsify_type
    #     # else:
    #     #     assert args.sparsify_type in ['longformer', 'diffuser'], "sparsify_type doesn not exist!"
    #     #     trainer_kwargs['sparsify_type'] = args.sparsify_type

    #     trainer = MyTrainer(**trainer_kwargs)
    # else:
    #     from transformers import Trainer
    #     trainer = Trainer(**trainer_kwargs)    

    t0_train = time()  # record train time    
    trainer.train(ignore_keys_for_eval=["hidden_states", "attentions", "global_attentions"])  # "loss"
    #trainer.train()
    train_secs = time() - t0_train

    if master_process:
        # get performance history
        if len(trainer.state.log_history) >= 1:
            run_perf = convert_train_history(trainer.state.log_history[:-1])
            col_names = list(run_perf.columns)
            top_names = ['epoch', 'step', 'learning_rate']
            top_names += [e for e in col_names if e not in top_names]
            run_perf = run_perf[top_names]
            run_perf.to_csv(njoin(model_dir, "run_performance.csv"))
        
        model_settings = attn_setup # model_settings['sparsify_type'] = args.sparsify_type
        model_settings['train_secs'] = train_secs
        model_settings.update(trainer.state.log_history[-1])
        final_perf = pd.DataFrame()
        final_perf = final_perf.append(model_settings, ignore_index=True)    
        final_perf.to_csv(njoin(model_dir, "final_performance.csv"))
        train_settings = pd.DataFrame(columns=["lr", "lr_scheduler_type", "train_bs", "eval_bs",
                                               "epochs", "weight_decay", "eval_strat", "eval_steps",
                                               "log_strat", "logging_steps", "save_steps",                          
                                               "seed", "warmup_steps",  "grad_accum_step", 
                                               "milestones", "gamma"], index=range(1))
        train_settings.iloc[0] = [args.lr, args.lr_scheduler_type, args.train_bs, args.eval_bs,
                                  args.epochs, args.weight_decay, args.eval_strat, args.eval_steps,
                                  args.log_strat, args.logging_steps, args.save_steps, 
                                  args.seed, args.warmup_steps, args.grad_accum_step, args.milestones,
                                  args.gamma]
        train_settings.to_csv(njoin(model_dir, "train_setting.csv"))

        print('\n')
        print('---------- Model trained and saved! ----------')