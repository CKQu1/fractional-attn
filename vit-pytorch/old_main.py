import argparse
import numpy as np
import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from time import time, sleep
from typing import Union
from constants import DROOT, MODEL_NAMES
from mutils import njoin, create_model_dir, convert_train_history, structural_model_root

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer

from tqdm import tqdm
from os import makedirs
from os.path import isdir, isfile
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import AdamW
from transformers.utils import logging
from datasets import load_dataset, load_metric, load_from_disk

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)


from torch import nn, optim
from tutils import save_experiment, save_checkpoint
from data_utils import prepare_data


class MyTrainer:
    """
    The simple trainer.
    """

    def __init__(self, config, model, optimizer, loss_fn, exp_name, device, **kwargs):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.base_dir = kwargs.get('base_dir')

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in tqdm(range(epochs)):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1,
                                base_dir=self.base_dir)
        # Save the experiment
        save_experiment(self.exp_name, self.config, self.model, train_losses, test_losses, accuracies,
                        base_dir=self.base_dir)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


# quick run (single unit)
"""
python -i main.py --epochs=1

python -i main.py --epochs=1 --use_faster_attn=False

python -i main.py --epochs=1 --model_name=fnsvit 

python -i main.py --epochs=1 --model_name=fnsvit --use_faster_attn=False
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='vit-pytorch/main.py training arguments')    
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--train_bs', default=4, type=int)
    parser.add_argument('--eval_bs', default=10, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--max_steps', default=None, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    # parser.add_argument('--milestones', default='', type=str or list) # Epoch units
    # parser.add_argument('--gamma', default=0.1, type=float) # Decay factor    

    # Log options
    # parser.add_argument('--eval_strat', default='steps', type=str)
    # parser.add_argument('--eval_steps', default=200, type=int)
    # parser.add_argument('--log_strat', default='steps', type=str)
    # parser.add_argument('--logging_steps', default=50, type=int)
    # parser.add_argument('--save_steps', default=50, type=int)
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--warmup_steps', default=2, type=int)
    # parser.add_argument('--grad_accum_step', default=8, type=int)
    # parser.add_argument('--debug', default=False, type=bool)  # for debuggin
    # parser.add_argument('--lr_scheduler_type', default='constant', type=str)
    # parser.add_argument('--do_train', default=True, type=bool)
    # parser.add_argument('--do_eval', default=True, type=bool)
    parser.add_argument('--model_root', default=njoin(DROOT, 'trained_models'), type=str, help='root dir of storing the model')

    # Model settings    
    # parser.add_argument('--sparsify_type', default=None, type=str)
    parser.add_argument('--qk_share', default=False, type=bool)
    parser.add_argument('--model_name', default='dpvit', type=str, help='dpvit | fnsvit')    
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)  

    # Dataset settings
    parser.add_argument('--dataset_name', default='cifar10', type=str)
    parser.add_argument('--divider', default=1, type=int)  # downsizing the test dataset    

    # config settings
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--hidden_size', default=48, type=int)
    #parser.add_argument('--intermediate_size', default=4 * 48, type=int)    
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_attn_heads', default=2, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.0, type=float)
    parser.add_argument('--attention_probs_dropout_prob', default=0.0, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--qkv_bias', default=True, type=bool)
    parser.add_argument('--use_faster_attn', default=True, type=bool)
    
    parser.add_argument("--save_model_every", default=0, type=int)
    parser.add_argument("--exp_name", default='image-task', type=str)

    args = parser.parse_args()    
    assert args.model_name in MODEL_NAMES, 'model_name does not exist!'

    # ---------- Device and system ----------
    dev = torch.device(f"cuda:{torch.cuda.device_count()}"
                       if torch.cuda.is_available() else "cpu")           

    # ---------- Model config ----------
    config = {
        "patch_size": args.patch_size,  # Input image size: 32x32 -> 8x8 patches
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.n_layers,
        "num_attention_heads": args.n_attn_heads,
        "intermediate_size": 4 * args.hidden_size, # 4 * hidden_size
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "initializer_range": args.initializer_range,
        "image_size": args.image_size,
        "num_classes": args.n_classes, # num_classes of CIFAR10
        "num_channels": args.n_channels,
        "qkv_bias": args.qkv_bias,
        "use_faster_attention": args.use_faster_attn,
    }
    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']
    assert config['image_size'] % config['patch_size'] == 0       

    attn_setup = {'qk_share': args.qk_share}
    attn_setup['model_name'] = args.model_name
    attn_setup['dataset_name'] = args.dataset_name
    if args.model_name == 'fnsvit':
        config['beta'] = attn_setup['beta'] = args.beta      
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth   
        if args.beta < 2:            
            config['d_intrinsic'] = int(config["hidden_size"] / config["num_attention_heads"])  # head_dim
            config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic'])-1)/np.pi)                            
        else:
            config['sphere_radius'] = 1

        attn_setup['sphere_radius'] = config['sphere_radius']       
        attn_setup['mask_val'] = np.pi * config['sphere_radius']               

    batch_size = args.train_bs
    epochs = args.epochs    
    save_model_every_n_epochs = args.save_model_every

    # ---------- Load dataset ----------
    if args.dataset_name == 'cifar10':
        trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    
    # ---------- Create the model, optimizer, loss function and trainer ----------
    if args.model_name == 'dpvit':
        from vit_pytorch.vit import ViTForClassfication
        model = ViTForClassfication(config)
    elif args.model_name == 'fnsvit':
        from vit_pytorch.fns_vit import FNSViTForClassfication
        model = FNSViTForClassfication(config)        

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
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
    models_dir, model_dir = create_model_dir(model_root, **attn_setup)

    #quit()  # delete
    trainer = MyTrainer(config, model, optimizer, loss_fn, args.exp_name, device=dev, base_dir=model_dir)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)    