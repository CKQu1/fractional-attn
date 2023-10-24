import argparse
import numpy as np
import os
import pandas as pd
import uuid
import torch
from time import time
from path_setup import droot

from os.path import join
from sklearn.metrics import f1_score
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import RobertaTokenizer
from transformers.utils import logging
from datasets import load_dataset,load_metric,load_from_disk
from models.diffuser_app import DiffuserForSequenceClassification
from models.diffuser_utils import DiffuserConfig
from graphtrainer import graphTrainer

dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                   if torch.cuda.is_available() else "cpu")  
print(f"Device in use: {dev}.")

# for enumerating each instance of training
def get_instance(dir, s):
    instances = []
    for dirname in next(os.walk(dir))[1]:        
        if s in dirname and len(os.listdir(join(dir, dirname))) > 0:
            try:                
                instances.append(int(dirname.split(s)[-1]))
            except:
                pass        
    return max(instances) + 1 if len(instances)>0 else 0
    
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
    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb', type=str)    

    args = parser.parse_args()    
    train_with_ddp = torch.distributed.is_available() and args.train_with_ddp

    repo_dir = os.getcwd()  # main dir    

    logging.set_verbosity_debug()
    logger = logging.get_logger()

    # should max_length also be added to parser args?
    max_length = args.max_length
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        #labels = eval_pred.label_ids
        predictions = eval_pred.predictions.argmax(1)
        #predictions = np.argmax(predictions, axis=1)
        #predictions = predictions[:, 0]
        #predictions = np.argmax(predictions, axis=-1)
        # local scripts
        metric_acc = load_metric(f'{repo_dir}/metrics/accuracy')
        metric_f1 = load_metric(f'{repo_dir}/metrics/f1')        
        #acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        #f1_score = metric_f1.compute(predictions=predictions, references=labels)["f1"]                   
        acc = metric_acc.compute(predictions=predictions, references=labels)
        f1_score = metric_f1.compute(predictions=predictions, references=labels)
        return {"accuracy": acc, "f1_score": f1_score }

    """
    # from load_metric()
    #metric_acc = load_metric("accuracy")
    #metric_f1 = load_metric("f1")
    # ----- additional metrics -----           
    #metric_prcn = load_metric("precision") 
    #metric_recall = load_metric("recall")   
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        #preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        preds = np.argmax(preds, axis=1)
        labels = eval_pred.label_ids
        #acc = metric_acc.compute(predictions=preds, references=labels)
        acc = (preds == p.labels).astype(np.float32).mean().item()
        #f1_score = metric_f1.compute(predictions=preds, references=labels)

        #return {"accuracy": acc, "f1_score": f1_score }
        return {"accuracy": acc}
    """

    """
    from sklearn.metrics import accuracy_score
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions.argmax(axis=1)
        labels = eval_pred.label_ids
        accuracy = accuracy_score(eval_pred.label_ids, eval_pred.predictions.argmax(axis=-1))
        return {'accuracy': accuracy}    
        #return accuracy
    """

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
        print("Downloading data!")
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.map(remove_columns=["text"])
        os.makedirs(dataset_dir)
        tokenized_dataset.save_to_disk(dataset_dir)
    else:
        print("Data downloaded, loading from local now!")
        tokenized_dataset = load_from_disk(dataset_dir)

    # ---------- REMOVE LATER ----------  
    divider = 500  
    tokenized_dataset['test'] = tokenized_dataset['test'].filter(lambda example, idx: idx % divider == 0, with_indices=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config.json")
    config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config_simple.json")
    config.num_labels = 2
    with_frac = args.with_frac
    attn_setup = {"with_frac":with_frac}
    print(f"with_frac: type {type(with_frac)} value {with_frac}")
    if with_frac:
        attn_setup["gamma"] = args.gamma     
    model =  DiffuserForSequenceClassification(config, **attn_setup).to(dev) 

    #if_create_model = not train_with_ddp
    #if train_with_ddp:
    #    if_create_model = if_create_model or (torch.distributed.get_rank() == 0)
    #if (not train_with_ddp) or (train_with_ddp and torch.distributed.get_rank() == 0):        
    if args.model_dir == None:                
        model_root_dir = join(droot, "trained_models", "seq_classification")            
    else:
        model_root_dir = args.model_dir
    if with_frac:            
        model_root_dir = join(model_root_dir, f"save_{args.dataset_name}_frac_diffuser_test")
    else:            
        model_root_dir = join(model_root_dir, f"save_{args.dataset_name}_diffuser_test")
    if not os.path.isdir(model_root_dir): os.makedirs(model_root_dir)    
    instance = get_instance(model_root_dir, "model_")
    #uuid_ = str(uuid.uuid4())[:8]  # for labelling training instance
    model_dir = join(model_root_dir, f"model_{instance}")
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
        device_name, device_total = "GPU", torch.cuda.device_count()
    else:
        device_name, device_total = "CPU", torch.get_num_threads()

    if not train_with_ddp:
        device_total = 1
    steps_per_train_epoch = int(len(tokenized_dataset['train'])/(training_args.per_device_train_batch_size*device_total*training_args.gradient_accumulation_steps ))
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
        #compute_metrics = compute_metrics
    )

    t0_train = time()  # record train time
    trainer.train()
    train_secs = time() - t0_train

    model_settings = attn_setup
    model_settings['train_secs'] = train_secs
    for key_name in model_settings.keys():
        model_settings[key_name] = [model_settings[key_name]]
    df = pd.DataFrame(model_settings)
    df.to_csv(join(model_dir, "model_settings.csv"))

    # get performance history
    #df_model = pd.DataFrame(trainer.state.log_history)
    df_model = convert_train_history(trainer.state.log_history)
    df_model.to_csv(join(model_dir, "model_performance.csv"))

    # save final model
    trainer.save_model()    