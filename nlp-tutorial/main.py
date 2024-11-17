import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
from torch.utils.data import DataLoader

from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
from trainer import Trainer

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

######################################################

import json
import os
import platform
import pandas as pd
import time
import torch.nn as nn
import math
import pickle
from contextlib import nullcontext
from os.path import isdir, isfile
from os import makedirs
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#from model import GPTConfig, GPT

from constants import *
from mutils import njoin, create_model_dir, convert_train_history, structural_model_root
from mutils import str2bool, str2ls

#from torch.optim import AdamW
from torch.optim import Adam

# single-core
"""
python -i main.py --train_bs=32 --dataset_name=cifar10 --model_name=opdpvit --qk_share=True\
 --epochs=1 --weight_decay=0 --n_layers=2 --n_attn_heads=1 --model_root=.droot/debug-mode 
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/main.py training arguments')   
    # training settings 
    #parser.add_argument('--train_with_ddp', default=True, type=bool, help='to use DDP or not')
    parser.add_argument('--max_iters', default=10, type=int)
    #parser.add_argument('--sub', default=1.0, type=float)
    parser.add_argument('--grad_clip', default=0, type=float)
    parser.add_argument('--decay_lr', default=True, type=bool)
    parser.add_argument('--warmup_iters', default=0, type=int)
    #parser.add_argument('--grad_accum_step', default=8, type=int)

    parser.add_argument('--max_lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min learning rate')
    parser.add_argument('--train_bs', default=2, type=int)
    parser.add_argument('--max_len',    default=512,  type=int,   help='the maximum size of the input sequence')

    #parser.add_argument('--eval_bs', default=10, type=int)
    #parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)       

    parser.add_argument('--lr_scheduler_type', default='cosine', type=str, help='cosine | binary | constant') 
    
    # log settings
    parser.add_argument('--epochs', default=None)

    parser.add_argument('--eval_interval', default=5, type=int)
    parser.add_argument('--log_interval', default=5, type=int)
    parser.add_argument('--eval_iters', default=200, type=int)
    parser.add_argument('--eval_only', default=False, type=bool)
    parser.add_argument('--always_save_checkpoint', default=True, type=bool)    
    parser.add_argument('--model_root', default='', type=str, help='root dir of storing the model')
    
    parser.add_argument("--save_model_every", default=0, type=int)
    parser.add_argument("--exp_name", default='image-task', type=str)

    parser.add_argument('--instance', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)    
    # parser.add_argument('--lr_scheduler_type', default='constant', type=str)

    # Tokenizer
    parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--tokenizer_name', default='sentencepiece', type=str)  
    parser.add_argument('--pretrained_model',    default='wiki.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')

    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb', type=str)
    parser.add_argument('--cache_dir', default=njoin(DROOT, 'cache_dir'), type=str)  

    # Model settings
    parser.add_argument('--model_name', default='spfnsvit', type=str)  
    # fnsvit type
    parser.add_argument('--manifold', default='sphere', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)  
    parser.add_argument('--a', default=0, type=float)
    # sinkvit type
    parser.add_argument('--n_it', default=3, type=int)

    # Config settings    
    # parser.add_argument('--hidden_size', default=48, type=int)
    # parser.add_argument('--n_layers', default=1, type=int)
    # parser.add_argument('--n_attn_heads', default=2, type=int)
    # #parser.add_argument('--intermediate_size', default=4 * 48, type=int)

    parser.add_argument('--hidden',         default=256,  type=int,   help='the number of expected features in the transformer')
    parser.add_argument('--n_layers',       default=1,    type=int,   help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_attn_heads',   default=4,    type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--dropout',        default=0.1,  type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden',     default=256, type=int,   help='the dimension of the feedforward network')    

    parser.add_argument('--qkv_bias', type=str2bool, nargs='?', const=True, default=False) 
    parser.add_argument('--qk_share', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--is_op', type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()    

    # assertions
    model_name = args.model_name.lower()
    if 'fns' in model_name:
        assert args.manifold in ['sphere', 'rd'], 'FNS manifold: sphere or rd'   
        assert 1 <= args.alpha <= 2, 'FNS alpha must be between [1,2]'
        assert args.a in [0,0.5,1], 'Normalization index must be 0 or 0.5 or 1'   

    # -----------------------------------------------------------------------------

    eval_interval = args.eval_interval
    log_interval = args.log_interval
    eval_iters = args.eval_iters
    eval_only = args.eval_only # if True, script exits right after the first eval
    always_save_checkpoint = args.always_save_checkpoint # if True, always save a checkpoint after each eval

    # data
    dataset_name = args.dataset_name
    batch_size = args.train_bs # if gradient_accumulation_steps > 1, this is the micro-batch size     

    # adamw optimizer
    learning_rate = args.max_lr  # max learning rate
    max_iters = args.max_iters # total number of training iterations
    weight_decay = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2
    grad_clip = args.grad_clip # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = args.decay_lr # whether to decay the learning rate
    warmup_iters = args.warmup_iters # how many steps to warm up for
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
    min_lr = args.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if 'cuda' in device else platform.processor()
    # -----------------------------------------------------------------------------        

    print('-'*25)
    print(f'device = {device}')
    print('-'*25 + '\n')           

    ##### SET SEED #####
    torch.manual_seed(args.seed)   
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    
    # poor man's data loader
    if dataset_name == 'imdb':
        # Load tokenizer
        if args.tokenizer_name == 'sentencepiece':
            tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
        else:
            tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
            tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file =args.vocab_file)
        # Build DataLoader
        train_dataset = create_examples(args, tokenizer, mode='train')
        test_dataset = create_examples(args, tokenizer, mode='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)        

    train_size = len(train_loader.dataset)
    eval_size = len(test_loader.dataset)                      
    steps_per_epoch = len(train_loader)

    epochs = args.epochs
    if epochs is not None:  
        epochs = int(epochs)
        max_iters = steps_per_epoch * epochs    
        eval_interval = steps_per_epoch
        log_interval = steps_per_epoch  # for mfu
    
    else:
        max_iters = args.max_iters
        eval_interval = args.eval_interval
        log_interval = args.log_interval
        eval_iters = args.eval_iters  

    def get_batch(split):
        if split == 'train':
            data = train_loader
        else:
            data = test_loader
        ix = torch.randint(len(data), (batch_size,))
        x = torch.stack([data.dataset[i][0] for i in ix])
        y = torch.tensor([data.dataset[i][1] for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y        

    # init a new model from scratch
    print(f'Initializing a new {model_name} from scratch \n')

    # These are not hard constraints, but are used to prevent misconfigurations
    assert args.hidden % args.n_attn_heads == 0    
    config = {
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "n_heads": args.n_attn_heads,
        "head_dim": args.hidden//args.n_attn_heads,
        "d_model": args.hidden,
        "d_ff": args.ffn_hidden, 
        "p_drop": args.dropout,  
        "vocab_size": tokenizer.vocab_size, 
        "seq_len": args.max_len,
        "pad_id": tokenizer.pad_token_id,
        "qkv_bias": args.qkv_bias,
        "qk_share": args.qk_share,     
        "is_op":    args.is_op        
    }

    attn_setup = {'qk_share': args.qk_share, 'qkv_bias': args.qkv_bias, 'is_op': args.is_op,
                  'instance': args.instance,
                  'dataset_name': args.dataset_name}        
    
    if 'fns' in model_name:
        attn_setup['manifold'] = args.manifold
        config['alpha'] = attn_setup['alpha'] = args.alpha      
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth          
        if args.manifold == 'sphere':

            if args.alpha < 2:
                if args.n_attn_heads == 1:
                    config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden//args.n_attn_heads - 1
                    config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic'])-1)/np.pi)   
                    #config.sphere_radius = 1   
                else:
                    config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden//args.n_attn_heads
                    config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic']))/np.pi)                                   
            elif args.alpha >= 2:
                config['sphere_radius'] = attn_setup['d_intrinsic'] = 1                 
        
            # mask for distance
            config['mask_val'] = attn_setup['mask_val'] = config['sphere_radius'] * np.pi
            attn_setup['sphere_radius'] = config['sphere_radius']   

            model_name = 'sp' + model_name

        elif args.manifold == 'rd':
            if args.alpha < 2:
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden//args.n_attn_heads  # head_dim                

            model_name = 'rd' + model_name

        # degree index
        config['a'] = attn_setup['a'] = args.a      

    elif 'sink' in model_name:
        config['n_it'] = attn_setup['n_it'] = args.n_it
        #config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth

    if model_name == 'rdfnsformer':
        from v2_models.rdfnsformer import RDFNSformer
        model = RDFNSformer(config)         
    elif model_name == 'spfnsformer':
        from v2_models.spfnsformer import SPFNSformer
        model = SPFNSformer(config)                                      
    elif model_name == 'sinkformer':
        from v2_models.sinkformer import SINKformer
        model = SINKformer(config)
    elif model_name == 'dpformer':
        from v2_models.dpformer import DPformer
        model = DPformer(config)                   

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of parameters of the model is %d' % count_parameters(model))

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    model = torch.nn.DataParallel(model)
    model.to(device)

    model_name = 'op' + model_name if args.is_op else model_name
    attn_setup['model_name'] = model_name   

    if args.model_root == '':
        model_root = structural_model_root(qk_share=args.qk_share, n_layers=args.n_layers,
                                           n_attn_heads=args.n_attn_heads, hidden=args.hidden  # lr=args.lr, bs=args.train_bs,                                                                                          
                                           )       
        model_root = njoin(DROOT, model_root)
    else:
        model_root = args.model_root  
    models_dir, out_dir = create_model_dir(model_root, **attn_setup)   
    
    os.makedirs(out_dir, exist_ok=True)  # makedir of out_dir

    # save config
    with open(njoin(out_dir,"config.json"), "w") as ofile: 
        json.dump(config, ofile)   
    # save attn_setup
    with open(njoin(out_dir,"attn_setup.json"), "w") as ofile: 
        json.dump(attn_setup, ofile)   

    # save train settings
    train_settings = pd.DataFrame(columns=["max_lr", "min_lr", "batch_size", "beta1", "beta2",
                                            "train_size", "eval_size", "steps_per_epoch",
                                            "max_iters", "weight_decay", "grad_clip", "decay_lr",
                                            "lr_scheduler_type",
                                            "eval_interval", "log_interval", "eval_iters", "eval_only", "always_save_checkpoint",                         
                                            "warmup_iters",
                                            "device_name"
                                            ], index=range(1))
    train_settings.iloc[0] = [args.max_lr, args.min_lr, args.train_bs, args.beta1, args.beta2,
                              train_size, eval_size, steps_per_epoch,
                              max_iters, args.weight_decay, args.grad_clip, args.decay_lr,
                              args.lr_scheduler_type,
                              eval_interval, log_interval, eval_iters, args.eval_only, args.always_save_checkpoint,
                              args.warmup_iters,
                              device_name
                              ]
    train_settings.to_csv(njoin(out_dir, "train_setting.csv"))           

    #scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    #optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)    
    optimizer = Adam(model.parameters(), lr=learning_rate)  # sinkformer
    #optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1,beta2), weight_decay=args.weight_decay)    

    # helps estimate an arbitrarily accurate loss over either split using many batches
    """
    @torch.no_grad()
    def estimate_loss():
        out_loss = {}
        out_acc = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            accs = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                #X, Y, _ = get_batch(split)
                with ctx:
                    #logits, loss = model(X, Y)
                    logits, _ = model(X)
                    loss = loss_fn(logits, Y)
                    predictions = torch.argmax(logits, dim=1)
                    correct = torch.sum(predictions == Y).item() / len(Y)
                losses[k] = loss.item()
                accs[k] = correct
            out_loss[split] = losses.mean()
            out_acc[split] = accs.mean()
        model.train()
        return out_loss, out_acc
    """

    @torch.no_grad()
    def fb_estimate_val():
        model.eval()
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for batch in test_loader:    

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

        return epoch_val_accuracy, epoch_val_loss

    # learning rate decay scheduler (cosine with warmup)    
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)    
    
    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    #X, Y, IX = get_batch('train')
    #IXs = IX
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    #raw_model = model.module if ddp else model # unwrap DDP container if needed
    raw_model = model
    running_mfu = -1.0

    metrics_ls = []    

    t0 = time.time()
    dt = None
    iter_num = 0
    best_val_loss = 1e9

    metric_cols = ['iter', 'lr', 'train_loss', 'val_loss', 'train_acc', 'val_acc','secs_per_eval']
    train_n_batches, train_n_samples = len(train_loader), len(train_loader.dataset)    

    prev_lr = None
    for epoch in range(epochs):    
    #for epoch in tqdm(range(epochs)):
        print('epoch: ', epoch)
        epoch_loss = 0
        epoch_accuracy = 0

        for batch in tqdm(train_loader):

            inputs, labels = batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)            

            outputs, attention_weights = model(inputs)
            
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            acc = (outputs.argmax(dim=-1) == labels).sum()
            epoch_accuracy += acc.item()            

            # determine and set the learning rate for this iteration
            if args.lr_scheduler_type == 'cosine':
                lr = get_lr(iter_num) if decay_lr else learning_rate
            elif args.lr_scheduler_type == 'binary':
                #if iter_num < max_iters * 2/3:
                # if iter_num < max_iters * 4/5:
                #     lr = learning_rate
                # else:
                #     lr = min_lr        
                if epoch + 1 < 15:
                    lr = learning_rate
                else:
                    lr = min_lr
            elif args.lr_scheduler_type == 'constant':
                lr = learning_rate
            if lr != prev_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr      

            optimizer.zero_grad()
            loss.backward()
            # clip the gradient
            if grad_clip != 0.0:                
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # if iter_num == 0 and eval_only:
            #     break

            iter_num += 1
            prev_lr = lr

        epoch_loss /= train_n_batches
        epoch_accuracy /= train_n_samples

        epoch_val_accuracy, epoch_val_loss = fb_estimate_val()

        # evaluate the loss on train/val sets and write checkpoints            
        metrics_ls.append([iter_num, lr, epoch_loss, epoch_val_loss, epoch_accuracy, epoch_val_accuracy, dt])

        df = pd.DataFrame(metrics_ls, columns=metric_cols)
        df.to_csv(njoin(out_dir, '_run_performance.csv'))               

        #if epoch_val_loss < best_val_loss or always_save_checkpoint:
        if always_save_checkpoint:
            best_val_loss = epoch_val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    #'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )        

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

    if isfile(njoin(out_dir, '_run_performance.csv')):
        os.remove(njoin(out_dir, '_run_performance.csv'))
    df = pd.DataFrame(metrics_ls, columns=metric_cols)
    df.to_csv(njoin(out_dir, 'run_performance.csv'))

    print(f'All data saved under {out_dir}')        

    # -----------------------------------------------

    def train(self, epoch):
        t = time.time()
        if epoch == 12:
            for g in self.optimizer.param_groups:
                g['lr'] /= 10
        losses, accs = 0, 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        
        self.model.train()
        attention_weights_cpu = []

        for i, batch in enumerate(self.train_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            # |inputs| : (batch_size, seq_len), |labels| : (batch_size)

            outputs, attention_weights = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            losses += loss.item()
            acc = (outputs.argmax(dim=-1) == labels).sum()
            accs += acc.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % (n_batches//5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f} Acc: {:4f}%'.format(
                    i, i, n_batches, losses/i, accs/(i*self.args.batch_size)*100.))
        print(time.time() - t)
        losses_b = losses/n_batches
        acc_ns = accs/n_samples * 100.
        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses_b, acc_ns))
        return losses_b, acc_ns, attention_weights_cpu

    def validate(self, epoch):
        losses, accs = 0, 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                inputs, labels = map(lambda x: x.to(self.device), batch)
                # |inputs| : (batch_size, seq_len), |labels| : (batch_size)

                outputs, attention_weights = self.model(inputs)
                # |outputs| : (batch_size, 2), |attention_weights| : [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
                
                loss = self.criterion(outputs, labels)
                losses += loss.item()
                acc = (outputs.argmax(dim=-1) == labels).sum()
                accs += acc.item()

        losses_b = losses / n_batches
        acc_ns = accs / n_samples * 100.
        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses_b, acc_ns))
        return losses_b, acc_ns

    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        
        torch.save(self.model, path)    