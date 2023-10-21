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

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='main_seq_classification.py training arguments')    
    parser.add_argument('--train_with_ddp', default=False, type=str, help='to use DDP or not')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--evaluation_strategy', default="steps", type=str)
    parser.add_argument('--eval_steps', default=1, type=float)
    parser.add_argument('--logging_strategy', default="steps", type=str)
    parser.add_argument('--logging_steps', default=1, type=float)
    parser.add_argument('--save_steps', default=1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup_steps', default=2, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    # Model settings
    parser.add_argument('--with_frac', default=False, type=bool)
    parser.add_argument('--gamma', default=None, type=float)
    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb', type=str)

    args = parser.parse_args()    

    # main dir
    repo_dir = os.getcwd()

    logging.set_verbosity_debug()
    logger = logging.get_logger()

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=1024)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        f1_score = metric_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": acc, "f1_score": f1_score }

    metric_acc = load_metric(f'{repo_dir}/metrics/accuracy')
    metric_f1 = load_metric(f'{repo_dir}/metrics/f1')

    # create cache for dataset
    dataset_dir = join(droot, "DATASETS")
    if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)

    imdb = load_dataset("imdb", cache_dir=dataset_dir)
    tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                 vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                 merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                 max_length     = 1024)

    # save tokenized dataset
    dataset_dir = join(droot, "DATASETS", f"tokenized_{args.dataset_name}")
    if not os.path.isdir(dataset_dir): 
        print("Downloading data!")
        tokenized_dataset = imdb.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.map(remove_columns=["text"])
        os.makedirs(dataset_dir)
        tokenized_dataset.save_to_disk(dataset_dir)
    else:
        print("Data downloaded, loading from local now!")
        tokenized_dataset = load_from_disk(dataset_dir)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config.json")
    config = DiffuserConfig.from_json_file(f"{repo_dir}/models/config_simple.json")
    config.num_labels = 2
    with_frac = args.with_frac
    if with_frac:
        attn_setup = {"with_frac":with_frac, "gamma":args.gamma}
    else:
        attn_setup = {"with_frac":with_frac}
    model =  DiffuserForSequenceClassification(config, **attn_setup)

    train_with_ddp = torch.distributed.is_available() and args.train_with_ddp

    dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                        if torch.cuda.is_available() else "cpu")    
    model = model.to(dev)

    uuid_ = str(uuid.uuid4())[:8]    
    if with_frac:
        model_dir = join(droot, "trained_seq_classification", f"save_{dataset_name}_diffuser_test", uuid_)
    else:
        model_dir = join(droot, "trained_seq_classification", f"save_{dataset_name}_frac_diffuser_test", uuid_)    
    #model_dir = join(droot, "qsub_parser_test", uuid_)
    if not os.path.isdir(model_dir): os.makedirs(model_dir)
    
    training_args = TrainingArguments(
        output_dir = model_dir,
        learning_rate = args.lr,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        max_steps = args.max_steps,
        weight_decay = args.weight_decay,
        evaluation_strategy = args.evaluation_strategy,
        eval_steps = args.eval_steps,
        logging_strategy=args.logging_strategy,
        logging_steps = args.logging_steps,
        save_steps = args.save_steps,    
        seed = args.seed,
        warmup_steps = args.warmup_steps,
        gradient_accumulation_steps = args.gradient_accumulation_steps
    )

    if dev.type != "cpu":
        device_name, device_total = "GPU", torch.cuda.device_count()
    else:
        device_name, device_total = "CPU", torch.get_num_threads()

    if not train_with_ddp:
        device_total = 1
    steps_per_train_epoch = int(len(tokenized_dataset['train'])/(training_args.per_device_train_batch_size*device_total*training_args.gradient_accumulation_steps ))
    print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"{device_name} count: {device_total}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps} \n")    

    training_args.eval_steps    = int(steps_per_train_epoch)
    training_args.logging_steps = int(steps_per_train_epoch/5)
    training_args.save_steps    = int(steps_per_train_epoch)

    trainer = graphTrainer(
        model = model,
        config = config,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["test"],
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )

    t0_train = time()
    trainer.train()

    train_secs = time() - t0_train

    model_settings = attn_setup
    model_settings['train_secs'] = train_secs
    for key_name in model_settings.keys():
        model_settings[key_name] = [model_settings[key_name]]
    df = pd.DataFrame(model_settings)
    df.to_csv(join(model_dir, "model_settings.csv"))

    # get performance history
    df_model = pd.DataFrame(trainer.state.log_history)
    df_model.to_csv(join(model_dir, "model_performance.csv"))

    # save final model
    trainer.save_model()    