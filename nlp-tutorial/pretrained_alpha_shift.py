import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F  
from tqdm import tqdm
from os import makedirs
from os.path import isdir

from constants import HYP_CMAP, HYP_CNORM, FIGS_DIR, MODEL_SUFFIX
from mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from v2_models.rdfnsformer import RDFNSformer

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from data_utils import glove_create_examples

device = f'cuda' if torch.cuda.is_available() else "cpu"

alpha_new = 1.2
bandwidth_new = 0.01
model_dir = '.droot/debug-mode/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model=0/'

bandwidths = [1, bandwidth_new]

# ----- set config -----

attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
config['dataset_name'] = attn_setup['dataset_name']
config['max_len'] = config['seq_len']
main_args = AttrDict(config)

torch.manual_seed(attn_setup['instance'])  # set seed   

# ----- load dataset (tokenized by GloVe) -----

tokenizer = get_tokenizer("basic_english")
glove = GloVe(name='6B', dim=config['hidden'])
vocab_size = len(glove.stoi)   
train_dataset = glove_create_examples(main_args, tokenizer, mode='train')
test_dataset = glove_create_examples(main_args, tokenizer, mode='test')  
batch_size = int(train_setting.loc[0,'batch_size'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

# ----- load pretrained_model -----    

loss_fn = nn.CrossEntropyLoss()
checkpoint = njoin(model_dir, 'ckpt.pt')
#ckpt = torch.load(checkpoint)
ckpt = torch.load(checkpoint, map_location=device)

alpha_old = config['alpha']  # original alpha

losses, accs = [], []
print('Starting inference \n')
for alpha_idx, alpha in enumerate([alpha_old, alpha_new]):

    config['alpha'] = alpha
    config['bandwidth'] = bandwidths[alpha_idx]
    #model = RDFNSformer(config, is_return_dist=True)
    model = RDFNSformer(config)     
    model.load_state_dict(ckpt['model'])
    model.to(device)

    # ----- eval model -----

    # from main.py function fb_estimate_val
    model.eval()
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    for batch in tqdm(test_loader):    

        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _ = model(inputs)
        val_loss = loss_fn(outputs, labels)

        #val_acc = (val_logits.argmax(dim=1) == label).float().mean()
        val_acc = (outputs.argmax(dim=-1)==labels).sum()
        epoch_val_accuracy += val_acc.item()
        epoch_val_loss += val_loss.item() 
    epoch_val_loss /= len(test_loader)
    epoch_val_accuracy /= len(test_loader.dataset)    

    losses.append(epoch_val_loss)
    accs.append(epoch_val_accuracy)

    #print(f'Model alpha check: alpha = {model.layers[0].mha.fns_attn.alpha}')  # for verify

# ----- print results -----

print(
    f'Pretrained on alpha = {alpha_old}, shifted to {alpha_new}: \n loss from {losses[0]} to {losses[1]} \n acc from {accs[0]} to {accs[1]}'
)