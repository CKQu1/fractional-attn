import argparse
import numpy as np
import os
import pandas as pd
import torch
from time import time
from path_setup import droot

from os.path import join
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

# for enumerating each instance of training
def get_instance(dir, s):
    instances = []
    dirnames = next(os.walk(dir))[1]
    if len(dirnames) > 0:
        for dirname in dirnames:        
            if s in dirname and len(os.listdir(join(dir, dirname))) > 0:
                try:                
                    instances.append(int(dirname.split(s)[-1]))
                except:
                    pass        
        return max(instances) + 1 if len(instances)>0 else 0
    else:
        return 0
    
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

# debug model
#python -i main_seq_classification --max_steps=1 --logging_steps=1 --save_steps=1 --eval_steps=1 --warmup_steps=0 --gradient_accumulation_steps=1 
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
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--with_frac', default=False, type=bool)
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--uuid_', default="0", type=str)
    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb', type=str)    

    args = parser.parse_args()    

    dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                    if torch.cuda.is_available() else "cpu")      

    train_with_ddp = torch.distributed.is_available() and args.train_with_ddp
    global_rank = None
    if train_with_ddp:
        world_size =int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        print(f"global_rank: {global_rank}")    
        print(f"Device in use: {dev}.")

    repo_dir = os.getcwd()  # main dir    

    logging.set_verbosity_debug()
    logger = logging.get_logger()

    # should max_length also be added to parser args?
    max_length = args.max_length
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    
    def preprocess_logits_for_metrics(logits, labels):
        preds = logits.argmax(dim=-1)
        return preds
        
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
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

    # create cache for dataset
    dataset_dir = join(droot, "DATASETS")
    if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)
    dataset = load_dataset(args.dataset_name, cache_dir=dataset_dir)
    tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                 vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                 merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                 max_length     = max_length)
    # save tokenized dataset
    dataset_dir = join(droot, "DATASETS", f"tokenized_{args.dataset_name}")
    if not os.path.isdir(dataset_dir):         
        print("Downloading data!") if global_rank == 0 or not train_with_ddp else None
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.map(remove_columns=["text"])
        os.makedirs(dataset_dir)
        tokenized_dataset.save_to_disk(dataset_dir)
    else:        
        print("Data downloaded, loading from local now!") if global_rank == 0 or not train_with_ddp  else None
        tokenized_dataset = load_from_disk(dataset_dir)

    # ---------- REMOVE LATER ----------  
    divider = 25  
    tokenized_dataset['test'] = tokenized_dataset['test'].filter(lambda example, idx: idx % divider == 0, with_indices=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config.json")
    config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config_simple.json")
    config.num_labels = 2
    with_frac = args.with_frac
    attn_setup = {"with_frac":with_frac}
    if global_rank == 0 or not train_with_ddp:
        print("-"*25)
        print(f"with_frac = {with_frac} and gamma = {args.gamma}")
        print("-"*25 + "\n")
    if with_frac:
        attn_setup["gamma"] = args.gamma     
    model =  DiffuserForSequenceClassification(config, **attn_setup).to(dev) 
      
    if args.model_dir == None:                
        model_root_dir = join(droot, "trained_models", "seq_classification")            
    else:
        model_root_dir = args.model_dir
    if with_frac:            
        model_root_dir = join(model_root_dir, f"save_{args.dataset_name}_frac_diffuser_test")
    else:            
        model_root_dir = join(model_root_dir, f"save_{args.dataset_name}_diffuser_test")
    #if not train_with_ddp or (train_with_ddp and (global_rank==0)): 
    if not os.path.isdir(model_root_dir): os.makedirs(model_root_dir)    
    #instance = get_instance(model_root_dir, "model_")
    #model_dir = join(model_root_dir, f"model_{instance}")
    model_dir = join(model_root_dir, f"model_{args.uuid_}")
    #if not train_with_ddp or (train_with_ddp and (global_rank==0)): 
    if not os.path.isdir(model_dir): os.makedirs(model_dir)
    
    training_args_dict = {"output_dir": model_dir,
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
    
    if dev.type != "cpu":
        device_name = "GPU"
    else:
        device_name = "CPU"
    if not train_with_ddp:
        device_total = 1
    else:
        device_total = world_size
            
    steps_per_train_epoch = int(len(tokenized_dataset['train'])/(training_args.per_device_train_batch_size*device_total*training_args.gradient_accumulation_steps ))
    if global_rank == 0 or not train_with_ddp:
        print("-"*25)
        print(f"steps_per_train_epoch {steps_per_train_epoch}")
        print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        print(f"{device_name} count: {device_total}")
        print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        print("-"*25 + "\n")

    if isinstance(training_args.num_train_epochs, int) and args.max_steps == None:
        training_args.eval_steps    = int(steps_per_train_epoch)
        training_args.logging_steps = int(steps_per_train_epoch/5)
        training_args.save_steps    = int(steps_per_train_epoch)

    trainer = graphTrainer(
        use_dgl = not with_frac,
        model = model,
        config = config,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["test"],
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )

    t0_train = time()  # record train time
    trainer.train(ignore_keys_for_eval=["loss", "hidden_states", "attentions", "global_attentions"])
    train_secs = time() - t0_train

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
