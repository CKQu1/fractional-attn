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

logging.set_verbosity_debug()
logger = logging.get_logger()

def preprocess_function(examples):
    return tokenizer(examples['text'], padding = 'max_length', truncation=True, max_length = 1024)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = metric_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": acc, "f1_score": f1_score }

metric_acc = load_metric('./metrics/accuracy')
metric_f1 = load_metric('./metrics/f1')

# create cache for dataset
dataset_dir = join(os.getcwd(), droot, "DATASETS")
if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)

imdb = load_dataset("imdb", cache_dir=dataset_dir)
# tokenizer = RobertaTokenizer.from_pretrained("./roberta-tokenizer", max_length = 1024)
tokenizer = RobertaTokenizer(tokenizer_file = "./roberta-tokenizer/tokenizer.json",
                             vocab_file     = "./roberta-tokenizer/vocab.json",
                             merges_file    = "./roberta-tokenizer/merges.txt",
                             max_length     = 1024)

# save tokenized dataset
dataset_dir = join(droot, "DATASETS", "tokenized_imdb")
if not os.path.isdir(dataset_dir): 
    print("Downloading data!")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    tokenized_imdb = tokenized_imdb.map(remove_columns=["text"])
    os.makedirs(dataset_dir)
    tokenized_imdb.save_to_disk(dataset_dir)
else:
    print("Data downloaded, loading from local now!")
    tokenized_imdb = load_from_disk(dataset_dir)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#config = DiffuserConfig.from_json_file("./models/config.json")
config = DiffuserConfig.from_json_file("./models/config_simple.json")
config.num_labels = 2
with_frac = False
attn_setup = {"with_frac":with_frac}
model =  DiffuserForSequenceClassification(config, **attn_setup)

train_with_ddp = False
train_with_ddp = torch.distributed.is_available() and train_with_ddp

dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                    if torch.cuda.is_available() else "cpu")    
model = model.to(dev)

uuid_ = str(uuid.uuid4())[:8]
#model_dir = join(droot, "save_imdb_diffuser", uuid_)
model_dir = join(droot, "save_imdb_diffuser_test", uuid_)
if not os.path.isdir(model_dir): os.makedirs(model_dir)
training_args = TrainingArguments(
    output_dir = model_dir,
    learning_rate = 3e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    #num_train_epochs = 0.0025,
    #num_train_epochs = 0.00125,
    max_steps = 5,
    weight_decay = 0.01,
    evaluation_strategy = "steps",
    eval_steps = 1,
    logging_strategy="steps",
    #logging_steps = 500,
    #save_steps = 500,
    logging_steps = 1,
    save_steps = 1,    
    seed = 42,
    #warmup_steps = 50,
    warmup_steps = 2,
    #gradient_accumulation_steps = 8
    gradient_accumulation_steps = 1
    #prediction_loss_only=False
)

if dev.type != "cpu":
    device_name, device_total = "GPU", torch.cuda.device_count()
else:
    device_name, device_total = "CPU", torch.get_num_threads()

if not train_with_ddp:
    device_total = 1
# this is assuming DDP is being deployed?
steps_per_train_epoch = int(len(tokenized_imdb['train'])/(training_args.per_device_train_batch_size*device_total*training_args.gradient_accumulation_steps ))
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
    train_dataset = tokenized_imdb["train"],
    eval_dataset = tokenized_imdb["test"],
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