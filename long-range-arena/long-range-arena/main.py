"""Adapted from https://github.com/lindermanlab/S5/blob/main/s5/dataloaders/lra.py"""

import json
import argparse
import os
import pandas as pd
import time
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from os.path import isfile
import math

import numpy as np
import torch

from constants import *
from mutils import njoin, create_model_dir, structural_model_root

from torch.optim import AdamW

### CHANGES HERE ###
from models import ClassificationModel, RetrievalModel
from dataloading import Datasets
####################

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

# single-core
"""
# training based on steps
python -i ddp_main.py --model_name=dpnmt --num_heads=2\
 --max_iters=50 --eval_interval=10 --log_interval=10 --eval_iters=10 --weight_decay=0 --model_root=.droot/debug_mode

python -i ddp_main.py --model_name=sinknmt --num_heads=2\
 --max_iters=50 --eval_interval=10 --log_interval=10 --eval_iters=1 --weight_decay=0 --model_root=.droot/debug_mode 

# training based on epochs
python -i ddp_main.py --model_name=dpnmt --num_heads=2\
 --epochs=1 --weight_decay=0 --model_root=.droot/debug_mode

python -i ddp_main.py --model_name=fnsnmt --manifold=sphere --alpha=1.2 --num_heads=2\
 --max_iters=50 --eval_interval=10 --log_interval=10 --eval_iters=10 --weight_decay=0 --model_root=.droot/debug_mode 

python -i ddp_main.py --model_name=fnsnmt --alpha=1.2\
 --num_encoder_layers=2 --num_decoder_layers=2\
 --max_iters=50 --eval_interval=10 --log_interval=10 --eval_iters=10 --weight_decay=0 --model_root=.droot/debug_mode
"""

# multi-core
"""
torchrun --nnodes=1 --nproc_per_node=4 ddp_main.py --model_name=fnsnmt --alpha=1.5\
 --max_iters=50 --eval_interval=10 --log_interval=10 --eval_iters=200 --weight_decay=0 --model_root=.droot/multi-core 
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='long-range-arena/main.py training arguments')   
    # training settings 
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--max_iters', default=10, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--decay_lr', default=True, type=bool)
    parser.add_argument('--warmup_iters', default=0, type=int)

    parser.add_argument('--max_lr', default=6e-4, type=float, help='max learning rate')
    parser.add_argument('--min_lr', default=6e-5, type=float, help='min learning rate')
    parser.add_argument('--train_bs', default=2, type=int)
    parser.add_argument('--eval_bs', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)        
    
    # log settings
    parser.add_argument('--eval_interval', default=5, type=int) 
    parser.add_argument('--log_interval', default=5, type=int)  
    parser.add_argument('--eval_iters', default=200, type=int) 
    parser.add_argument('--eval_only', default=False, type=bool) 
    parser.add_argument('--always_save_checkpoint', default=True, type=bool) 
    parser.add_argument('--model_root', default=njoin(DROOT, 'trained_models'), type=str, help='root dir of storing the model')
    # CHANGES 
    parser.add_argument('--cache_dir', default=njoin(DROOT, 'cache_dir'), type=str)
    
    parser.add_argument("--init_from", default='scratch', type=str, help='scratch | resume | gpt2')
    parser.add_argument('--wandb_log', default=False, type=bool)
    parser.add_argument("--wandb_project", default='translation-task', type=str)    

    parser.add_argument('--instance', default='none', type=str)
    parser.add_argument('--seed', default=42, type=int)    
    # parser.add_argument('--debug', default=False, type=bool)  # for debuggin
    parser.add_argument('--lr_scheduler_type', default='constant', type=str)
    # parser.add_argument('--do_train', default=True, type=bool)
    # parser.add_argument('--do_eval', default=True, type=bool)

    # Model settings
    parser.add_argument('--model_name', default='fnsformer', type=str) # 
    # FNS
    parser.add_argument('--manifold', default='sphere', type=str) #
    parser.add_argument('--alpha', default=1, type=float) #
    parser.add_argument('--a', default=0, type=float) #
    # SINK
    parser.add_argument('--n_it', default=1, type=int)
    # general
    parser.add_argument('--bandwidth', default=1, type=float)
    parser.add_argument('--qk_share', default=False, type=bool)              
    parser.add_argument('--sphere_radius', default=1, type=float)  

    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb-classification', type=str)
    parser.add_argument('--divider', default=1, type=int)  # downsizing the test dataset    
    parser.add_argument('--tokenizer_path', default=None, type=str)  # tokenizer file path

    # Config settings
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--intermediate_size', default=128, type=int)    
    parser.add_argument('--num_encoder_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--num_classifier_layers', default=1, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.0, type=float)
    parser.add_argument('--encoder_dropout_prob', default=0.0, type=float)
    parser.add_argument('--attention_probs_dropout_prob', default=0.0, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)
    parser.add_argument('--qkv_bias', default=False, type=bool)
    parser.add_argument('--use_faster_attn', default=True, type=bool)
    parser.add_argument('--pooling_mode', default='MEAN', type=str)
    parser.add_argument('--interaction', default='none', type=str) # 'NLI' = NLI, else not
    
    # CHANGES HERE
    parser.add_argument('--apples_to_apples', default=False, type=bool)

    args = parser.parse_args()    
    
    # If apples_to_apples comparison with other models
    if args.apples_to_apples:
        if args.dataset_name == 'listops-classification':
            args.hidden_size = 512 # IMMUTABLE
            args.num_heads = 8 
            args.num_encoder_layers = 6 # IMMUTABLE
            args.intermediate_size = 2048 # IMMUTABLE
            args.pooling_mode = 'CLS' # Pooling not currently done in non-retrieval model
            args.max_iter = 5000 # Can probably change this
        elif args.dataset_name == 'imdb-classification':
            args.hidden_size = 512 # IMMUTABLE
            args.num_heads = 8 
            args.num_encoder_layers = 6 # IMMUTABLE
            args.intermediate_size = 2048 # IMMUTABLE
            args.pooling_mode = 'CLS'
            args.num_classifier_layers = 2 # 2-layer MLP
            args.max_iter = 20000 # Can probably change this
        elif args.dataset_name == 'aan-classification':
            args.hidden_size = 128 # IMMUTABLE
            args.num_heads = 4
            args.num_encoder_layers = 4 # IMMUTABLE
            args.intermediate_size = 512 # IMMUTABLE
            args.pooling_mode = 'CLS'
            args.interaction = 'NLI'
            args.max_iter = 5000 # Can probably change this
        elif args.dataset_name == 'lra-cifar-classification':
            args.hidden_size = 64 # IMMUTABLE
            args.num_heads = 4
            args.num_encoder_layers = 3 # IMMUTABLE
            args.intermediate_size = 128 # IMMUTABLE
            args.num_classifier_layers = 2 # 2-layer MLP
            # args.pooling_mode = 'CLS' # Not mentioned
            args.epochs = 200 # Can probably change this
            # Also did extensive hyperparameter sweeps
            

    # assertions
    model_name = args.model_name.lower()
    if 'fns' in model_name:
        assert args.manifold in ['sphere', 'rd'], 'FNS manifold: sphere or rd'   
        assert 1 <= args.alpha <= 2, 'FNS alpha must be between [1,2]'
        assert args.a in [0,0.5,1], 'Normalization index must be 0 or 0.5 or 1'             
    
    eval_interval = args.eval_interval
    log_interval = args.log_interval
    eval_iters = args.eval_iters
    eval_only = args.eval_only # if True, script exits right after the first eval
    always_save_checkpoint = args.always_save_checkpoint # if True, always save a checkpoint after each eval
    init_from = args.init_from
    # wandb logging
    wandb_log = args.wandb_log # disabled by default
    wandb_project = args.wandb_project
    wandb_run_name = f'{model_name}-{args.dataset_name}' # 'run' + str(time.time())

    # data
    dataset = args.dataset_name
    train_batch_size = args.train_bs # if gradient_accumulation_steps > 1, this is the micro-batch size
    eval_batch_size = args.eval_bs
    #block_size = 1024  # max sequence length (https://stackoverflow.com/questions/66294076/how-to-determine-the-block-size-in-training-a-dataset)

    # model
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    
    config = {
        "alpha": args.alpha,
        "a": args.a,
        "bandwidth": args.bandwidth,
        "sphere_radius": args.sphere_radius,
        "hidden_size": args.hidden_size,
        "num_encoder_layers": args.num_encoder_layers,
        "num_heads": args.num_heads,
        "intermediate_size": args.intermediate_size,
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "encoder_dropout_prob": args.encoder_dropout_prob,
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "initializer_range": args.initializer_range,
        "qkv_bias": args.qkv_bias,
        "use_faster_attention": args.use_faster_attn,
        "tokenizer_path": args.tokenizer_path,
        "pooling_mode": args.pooling_mode,
        "interaction": args.interaction,
    }
    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_heads"] == 0
    #assert config['intermediate_size'] == 4 * config['hidden_size']

    instance = int(args.instance) if args.instance.lower() != 'none' else None
    attn_setup = {'qk_share': args.qk_share, 'qkv_bias': args.qkv_bias}
    if instance is not None:
        attn_setup['instance'] = instance    
    attn_setup['dataset_name'] = args.dataset_name    
    if 'fns' in model_name:
        attn_setup['manifold'] = args.manifold
        config['alpha'] = attn_setup['alpha'] = args.alpha      
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth          
        if args.manifold == 'sphere':

            if args.alpha < 2:
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden_size//args.num_heads - 1
                config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic'])-1)/np.pi)   
                #config.sphere_radius = 1                
            elif args.alpha >= 2:
                config['sphere_radius'] = attn_setup['d_intrinsic'] = 1                 
        
            # mask for distance
            config['mask_val'] = attn_setup['mask_val'] = config['sphere_radius'] * np.pi
            attn_setup['sphere_radius'] = config['sphere_radius']   

            model_name = 'sp' + model_name

        elif args.manifold == 'rd':
            if args.alpha < 2:
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden_size//args.num_heads  # head_dim                

            model_name = 'rd' + model_name

        # degree index
        config['a'] = attn_setup['a'] = args.a       
    elif model_name == 'sinknmt':
        config['n_it'] = attn_setup['n_it'] = args.n_it
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth           

    attn_setup['model_name'] = model_name

    # adamw optimizer
    learning_rate = args.max_lr  # max learning rate
    epochs = args.epochs
    # previously had max_iters HERE
    weight_decay = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2
    grad_clip = args.grad_clip # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = args.decay_lr # whether to decay the learning rate
    warmup_iters = args.warmup_iters # how many steps to warm up for    
    min_lr = args.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    #device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    #device = f'cuda:{torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    #compile = True # use PyTorch 2.0 to compile the model to be faster
    compile = False
    # -----------------------------------------------------------------------------
    # CONFIGS FOR GPT2
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    
    if args.model_root == '':
        model_root = structural_model_root(qk_share=args.qk_share, num_encoder_layers=args.num_encoder_layers,
                                           num_decoder_layers=args.num_decoder_layers, num_heads=args.num_heads,
                                           hidden_size=args.hidden_size,
                                           lr=args.lr, bs=args.train_bs, 
                                           use_custom_optim=args.use_custom_optim,
                                           milestones=args.milestones, gamma=args.gamma,
                                           epochs=args.epochs                                               
                                           )       
        model_root = njoin(DROOT, 'ddp_formers', model_root)
    else:
        model_root = args.model_root  
    models_dir, out_dir = create_model_dir(model_root, **attn_setup)    
    
    # makedir of out_dir
    os.makedirs(out_dir, exist_ok=True)

    # save config
    with open(njoin(out_dir,"config.json"), "w") as ofile: 
        json.dump(config, ofile)   
    # save attn_setup
    with open(njoin(out_dir,"attn_setup.json"), "w") as ofile: 
        json.dump(attn_setup, ofile)                 
    # save train settings
    train_settings = pd.DataFrame(columns=["max_lr", "min_lr", "batch_size", "beta1", "beta2",
                                            "max_iters", "weight_decay", "grad_clip", "decay_lr",
                                            "lr_scheduler_type",
                                            "eval_interval", "log_interval", "eval_iters", "eval_only", "always_save_checkpoint",                         
                                            "warmup_iters"], index=range(1))
    train_settings.iloc[0] = [args.max_lr, args.min_lr, args.train_bs, args.beta1, args.beta2,
                                args.max_iters, args.weight_decay, args.grad_clip, args.decay_lr,
                                args.lr_scheduler_type,
                                args.eval_interval, args.log_interval, args.eval_iters, args.eval_only, args.always_save_checkpoint,
                                args.warmup_iters
                                ]
    train_settings.to_csv(njoin(out_dir, "train_setting.csv"))        

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # else:
    #     torch.backends.cpu.matmul.allow_tf32 = True # allow tf32 on matmul
    #     torch.backends.cpu.allow_tf32 = True # allow tf32 on cudnn        
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset_name]
    
    # Dataset dependent logic
    if dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False
    else:
        padded = False
        retrieval = False
        
    # Create dataset...
    trainloader, valloader, testloader, num_classes, seq_len, in_dim, train_size, vocab_size = \
        create_dataset_fn(cache_dir=args.cache_dir, seed=args.seed, train_bs=train_batch_size, eval_bs=eval_batch_size)
    # Add config parameters depending on dataset
    config["num_classes"] = num_classes 
    config["max_length"] = seq_len 
    config["vocab_size"] = vocab_size
    config["padding_idx"] = 0
    
    # Create fnsformer model; I don't anticipate we would need to run sinkformer
    # since its results on LRA are available.
    if init_from == 'scratch':
        print(f'Initializing a new {model_name} from scratch \n')
        # Initialize model from scratch
        if retrieval:
            from models import RetrievalModel
            print("Using RetrievalModel for {} task".format(dataset))
            model = RetrievalModel(config)
        else:
            from models import ClassificationModel
            print("Using ClassificationModel for {} task".format(dataset))
            model = ClassificationModel(config)
    model.to(device)
    
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1,beta2), weight_decay=args.weight_decay)
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model, backend='aot_eager')
    # logging
    if wandb_log:
        import wandb  # NOT IN CONTAINER
        wandb.init(project=wandb_project, name=wandb_run_name, config=config) 

    if epochs is None:
        steps_per_epoch = None
        max_iters = args.max_iters # total number of training iterations
    else:    
        epochs = int(epochs)
        steps_per_epoch = len(train_size) // train_batch_size + 1
        max_iters = epochs * steps_per_epoch
        eval_interval = steps_per_epoch
        log_interval = steps_per_epoch  # will need to change later
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla

    def get_batch(split):
        """Get batch from data iterator. Returns:
            (inputs, targets) if no aux data for lengths
            ((inputs, lengths), targets) if aux data for lengths
            inputs is B x seq_len"""
        if split == 'train':
            data = trainloader
        else:
            data = testloader # valloader is not used
        batch = next(iter(data))
        # Ignore aux data for lengths
        # Padding mask created below anyways
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, y, _ = batch
        # Pad to seq_len
        x = F.pad(x, (0,seq_len - x.shape[1],0,0), value=0) # padding_idx = 0
        x, y = x.to(device), y.to(device)
        # Create padding mask
        padding_mask = (x != 0).type(torch.int) # B x L
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).type(torch.int) # B x 1 x 1 x L
        return x, y, padding_mask
    
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

    # Training loop
    @torch.no_grad()
    def estimate_loss():
        out_loss = {}
        out_acc = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            accs = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, padding_mask = get_batch(split)
                with ctx:                    
                    logits, _ = model(X, attention_mask=padding_mask) # pre-softmax `logits`
                    loss = loss_fn(logits, Y) # CHECK
                    predictions = torch.argmax(logits, dim=1)
                    correct = torch.sum(predictions == Y).item() / len(Y)
                losses[k] = loss.item()
                accs[k] = correct
            out_loss[split] = losses.mean()
            out_acc[split] = accs.mean()
        model.train()
        return out_loss, out_acc
    
    t0 = time.time()   
    X, Y, padding_mask = get_batch('train')
    iter_num = 0 
    best_val_loss = 1e9
    dt = None
    
    if not wandb_log:
        metrics_ls = []    
    while True:
        # determine and set the learning rate for this iteration
        if args.lr_scheduler_type == 'cosine':
            lr = get_lr(iter_num) if decay_lr else learning_rate
        elif args.lr_scheduler_type == 'binary':
            if iter_num < max_iters * 2/3:
                lr = learning_rate
            else:
                lr = min_lr  
        elif args.lr_scheduler_type == 'constant':
            lr = learning_rate    
    
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses, acc = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train accuracy {acc['train']:.4f}, val accuracy {acc['val']:.4f}")
            if wandb_log: 
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "train/acc": acc['train'],
                    "val/acc": acc['val'],
                    "lr": lr
                    #"mfu": running_mfu*100, # convert to percentage
                })
            else:
                metrics_ls.append([iter_num, lr, losses['train'].item(), losses['val'].item(), acc['train'], acc['val'], dt])
                
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        #'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

            if not wandb_log:
                df = pd.DataFrame(metrics_ls, columns=['iter', 'lr', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'secs_per_eval'])
                df.to_csv(njoin(out_dir, '_run_performance.csv'))   
                
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1                                

        if iter_num == 0 and eval_only:
            break
        
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16        
        with ctx:
            ##### CHANGES HERE #####
            logits, _ = model(X, attention_mask=padding_mask)  # NOT REALLY LOGITS
            # logits (bs, seq_len, tgt_vocab_size)
            loss = loss_fn(logits, Y) 
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, padding_mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item()
            # if local_iter_num >= 5: # let the training loop settle a bit
            #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)  # mfu (model flop utilization)
            #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            #print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        
        # termination conditions
        if iter_num > max_iters:
            if not wandb_log:
                if isfile(njoin(out_dir, '_run_performance.csv')):
                    os.remove(njoin(out_dir, '_run_performance.csv'))
                df = pd.DataFrame(metrics_ls, columns=['iter', 'lr', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'secs_per_eval'])
                df.to_csv(njoin(out_dir, 'run_performance.csv'))
            print(f'All data saved under {out_dir}')
            break