import argparse
import numpy as np
import os
import pandas as pd
import torch
from time import time, sleep
from path_setup import droot

from os.path import join, isdir
from sklearn.metrics import f1_score
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import RobertaTokenizer
from transformers.utils import logging
from datasets import load_dataset, load_metric, load_from_disk
from models.diffuser_app import DiffuserForSequenceClassification
from models.diffuser_utils import DiffuserConfig
from graphtrainer import graphTrainer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings("ignore")
    
# convert trainer.state.log_history to df
def convert_train_history(ls):    
    df_model = pd.DataFrame()
    cur_dict = ls[0]
    cur_step = cur_dict['step']
    for idx, next_dict in enumerate(ls[1:]):        
        next_step = next_dict['step']
        if cur_step == next_step:
            cur_dict.update(next_dict)
        else:
            df_model = df_model.append(cur_dict, ignore_index=True)
            cur_dict = next_dict
            cur_step = cur_dict['step']
    df_model = df_model.append(cur_dict, ignore_index=True)
    return df_model

# simple wrapper for load_dataset
def get_dataset(dataset_name, cache_dir):
    if args.dataset_name in ['imdb', 'emotion', 'rotten_tomatoes']:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    elif args.dataset_name == "hyperpartisan_news_detection":
        dataset = load_dataset(dataset_name, "byarticle", cache_dir=cache_dir)
    elif args.dataset_name == "newsgroup":
        dataset = load_dataset(dataset_name, '18828_alt.atheism', cache_dir=cache_dir)   
    elif args.dataset_name == "cola":
        dataset = load_dataset('glue', 'cola', cache_dir=cache_dir) 
    else:
        raise NameError(f"{dataset_name} not applicable!")
    return dataset

# get dataset feature names
def get_dataset_cols(dataset):
    return list(dataset.column_names.values())[0]

# remove and rename key names in dataset
def process_dataset_cols(dataset):
    column_names = get_dataset_cols(dataset)
    if 'hyperpartisan' in column_names:
        dataset = dataset.rename_column("hyperpartisan", "label")
    if 'sentence' in column_names:
        dataset = dataset.rename_column('sentence', 'text')
    
    # for removing unused features i.e. text
    column_names = list(dataset.column_names.values())[0]
    remove_columns = list(set(column_names) - {'input_ids', 'attention_mask', 'label'})
    if len(remove_columns) > 0:
        dataset = dataset.map(remove_columns=remove_columns)
    
    return dataset

# quick run (single unit)
"""
python -i main_seq_classification.py  --with_frac=True --gamma=0.5 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=50 --warmup_steps=0 --gradient_accumulation_steps=1 --dataset_name=rotten_tomatoes\
 --model_dir=droot/debug_mode10/model_0
"""

# quick torchrun (multi-unit)
"""
singularity exec --home ${PBS_O_WORKDIR} ${cpath} torchrun --nproc_per_node=2\
 main_seq_classification.py --with_frac=True --gamma=0.75\
 --divider=1 --train_with_ddp=True\
 --gradient_accumulation_steps=4 --epochs=5 --warmup_steps=50\
 --per_device_train_batch_size=4 --per_device_eval_batch_size=4\
 --model_dir=${PBS_O_WORKDIR}/droot/new_unlapl_rot/frac_diffuser/model=0_gamma=0.75\
 --dataset_name=rotten_tomatoes
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='main_seq_classification.py training arguments')    
    parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--per_device_train_batch_size', default=2, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=10, type=int)
    parser.add_argument('--epochs', default=1, type=float)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--evaluation_strategy', default="steps", type=str)
    parser.add_argument('--eval_steps', default=200, type=int)
    parser.add_argument('--logging_strategy', default="steps", type=str)
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--save_steps', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup_steps', default=2, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    # Model settings    
    parser.add_argument('--model_name', default='diffuser', type=str)
    parser.add_argument('--with_frac', default=False, type=bool)
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--model_dir', default=None, type=str)
    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb', type=str)
    parser.add_argument('--divider', default=5, type=int)  # downsizing the test dataset

    args = parser.parse_args()    

    repo_dir = os.getcwd()  # main dir 
    dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                       if torch.cuda.is_available() else "cpu")   
    device_name = "GPU" if dev.type != "cpu" else "CPU"
    train_with_ddp = torch.distributed.is_available() and args.train_with_ddp
    global_rank = None
    if train_with_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        print(f"global_rank: {global_rank}")    
        print(f"Device in use: {dev}.")
        device_total = world_size
        #backend = "gloo" if dev.type != "cpu" else "nccl"
        #import torch.distributed as dist
        #dist.init_process_group(backend=backend)        
    else:
        device_total = 1       

    logging.set_verbosity_debug()
    logger = logging.get_logger()

    # ---------------------------------------- 1. Dataset setup ----------------------------------------

    # should max_length also be added to parser args?
    max_length = args.max_length    
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    def preprocess_logits_for_metrics(logits, labels):
        preds = logits.argmax(dim=-1)
        return preds
        
    if args.dataset_name == 'imdb':
        metric_acc = load_metric("accuracy")
        metric_f1 = load_metric("f1")
        #metric_prcn = load_metric("precision") 
        #metric_recall = load_metric("recall") 
    else:
        metric_acc = load_metric("accuracy", average='micro')
        metric_f1 = load_metric("f1", average='micro')
        #metric_prcn = load_metric("precision") 
        #metric_recall = load_metric("recall")         
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        acc = metric_acc.compute(predictions=preds, references=labels)        
        f1_score = metric_f1.compute(predictions=preds, references=labels)  
        #precision = metric_prcn.compute(predictions=preds, references=labels)        
        #recall = metric_recall.compute(predictions=preds, references=labels)                
        #return {"accuracy":acc,"f1_score":f1_score,"precision":precision,"recall":recall}
        return {"accuracy":acc,"f1_score":f1_score}
    
    if args.model_name == 'diffuser':
        tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                     vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                     merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                     max_length     = max_length)
    elif args.model_name == 'automodel':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # save tokenized dataset
    tokenized_dataset_dir = join(droot, "DATASETS", f"tokenized_{args.dataset_name}")
    if not os.path.isdir(tokenized_dataset_dir):         
        print("Downloading data!") if global_rank == 0 or not train_with_ddp else None
        # create cache for dataset
        dataset = get_dataset(args.dataset_name, join(droot, "DATASETS"))

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        column_names = get_dataset_cols(tokenized_dataset)
        if 'text' in column_names:
            tokenized_dataset = tokenized_dataset.map(remove_columns=['text'])
        if not isdir(tokenized_dataset_dir): os.makedirs(tokenized_dataset_dir)
        tokenized_dataset.save_to_disk(tokenized_dataset_dir)
        del dataset  # alleviate memory
    else:        
        print("Data downloaded, loading from local now! \n") if global_rank == 0 or not train_with_ddp else None
        tokenized_dataset = load_from_disk(tokenized_dataset_dir)
    if args.dataset_name != 'imdb':
        tokenized_dataset = process_dataset_cols(tokenized_dataset)

    keys = list(tokenized_dataset.keys())
    if len(keys) == 1:
        tokenized_dataset = tokenized_dataset[keys[0]].train_test_split(0.5)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]
    #del tokenized_dataset  # alleviate memory         

    # ---------- REMOVE LATER ----------  
    divider = args.divider  # downsize eval_dataset
    eval_dataset = eval_dataset.filter(lambda example, idx: idx % divider == 0, with_indices=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config.json")

    # ---------------------------------------- 2. Model setup ----------------------------------------

    config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config_simple.json")
    config.num_labels = len(set(train_dataset['label']))    
    attn_setup = {"with_frac": args.with_frac}
    if global_rank == 0 or not train_with_ddp:
        print("-"*25)
        print(f"with_frac = {args.with_frac} and gamma = {args.gamma}")
        print("-"*25 + "\n")  
    if args.model_name == 'diffuser':
        if args.with_frac:
            attn_setup["gamma"] = args.gamma        
        use_dgl = not args.with_frac
        model =  DiffuserForSequenceClassification(config, **attn_setup).to(dev)        
    elif args.model_name == 'automodel':
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased").to(dev)
    # can add other options here

    if args.model_dir == None:
        if args.with_frac and args.model_name == 'diffuser':
            args.model_dir = join(droot, "seq_classification", "save_{args.dataset_name}_sym_frac_diffuser")
        else:
            args.model_dir = join(droot, "seq_classification", "save_{args.dataset_name}_{args.model_name}")
    training_args_dict = {"output_dir": args.model_dir,
                          "learning_rate": args.lr,
                          "per_device_train_batch_size": args.per_device_train_batch_size,
                          "per_device_eval_batch_size": args.per_device_eval_batch_size,
                          "num_train_epochs": args.epochs,                          
                          "weight_decay": args.weight_decay,
                          "evaluation_strategy": args.evaluation_strategy,
                          "eval_steps": args.eval_steps,
                          "logging_strategy": args.logging_strategy,
                          "logging_steps": args.logging_steps,
                          "save_steps": args.save_steps,    
                          "seed": args.seed,
                          "warmup_steps": args.warmup_steps,
                          "gradient_accumulation_steps": args.gradient_accumulation_steps                          
                          }
    if args.max_steps != None:
        training_args_dict["max_steps"] = args.max_steps
    training_args = TrainingArguments(**training_args_dict)    
            
    steps_per_train_epoch = int(len(train_dataset)/(training_args.per_device_train_batch_size*device_total*training_args.gradient_accumulation_steps ))
    if global_rank == 0 or not train_with_ddp:
        print("-"*25)
        print(f"steps_per_train_epoch {steps_per_train_epoch}")
        print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        print(f"{device_name} count: {device_total}")
        print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        print("-"*25 + "\n")

    if training_args.num_train_epochs >= 1 and args.max_steps == None:
        training_args.eval_steps    = int(steps_per_train_epoch)
        #training_args.logging_steps = int(steps_per_train_epoch/5)
        training_args.logging_steps = int(steps_per_train_epoch/3)
        training_args.save_steps    = int(steps_per_train_epoch)

    trainer_kwargs = {'model': model,                      
                      'args': training_args,
                      'train_dataset': train_dataset,
                      'eval_dataset': eval_dataset,
                      'tokenizer': tokenizer,
                      'data_collator': data_collator,
                      'compute_metrics': compute_metrics,
                      'preprocess_logits_for_metrics': preprocess_logits_for_metrics
                      }
    if args.model_name == 'diffuser':
        trainer_kwargs['use_dgl'] = use_dgl
        trainer_kwargs['config'] = config
        trainer = graphTrainer(**trainer_kwargs)
    else:
        from transformers import Trainer
        trainer = Trainer(**trainer_kwargs)

    t0_train = time()  # record train time    
    trainer.train(ignore_keys_for_eval=["loss", "hidden_states", "attentions", "global_attentions"])
    train_secs = time() - t0_train

    model_dir = args.model_dir
    if not os.path.isdir(model_dir): os.makedirs(model_dir)
    # get performance history
    if len(trainer.state.log_history) > 1:
        run_perf = convert_train_history(trainer.state.log_history[:-1])
        col_names = list(run_perf.columns)
        top_names = ['epoch', 'step', 'learning_rate']
        top_names += [e for e in col_names if e not in top_names]
        run_perf = run_perf[top_names]
        run_perf.to_csv(join(model_dir, "run_performance.csv"))

    model_settings = attn_setup; model_settings['train_secs'] = train_secs
    model_settings.update(trainer.state.log_history[-1])
    final_perf = pd.DataFrame()
    final_perf = final_perf.append(model_settings, ignore_index=True)    
    final_perf.to_csv(join(model_dir, "final_performance.csv"))

    # save final model
    trainer.save_model(join(model_dir, "final_model"))