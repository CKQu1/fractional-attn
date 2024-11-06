import argparse
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
from data_utils import prepare_data

from torch.optim import AdamW

# single-core
"""
python -i main.py --train_bs=32 --dataset_name=cifar10 --model_name=opdpvit --qk_share=True\
 --epochs=1 --weight_decay=0 --n_layers=2 --n_attn_heads=1 --model_root=.droot/debug-mode 

python -i main.py --train_bs=32 --dataset_name=cifar10 --model_name=fnsvit --manifold=sphere --qk_share=True\
 --manifold=sphere --alpha=1.5 --a=0 --epochs=1 --weight_decay=0 --n_layers=2 --n_attn_heads=1\
 --model_root=.droot/debug-mode

python -i main.py --model_name=opfnsvit --manifold=rd --alpha=1.5 --max_iters=100 --eval_interval=5\
 --lr_scheduler_type=binary --max_lr=5e-5 --max_lr=5e-6\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2 --model_root=.droot/single-core  

python -i main.py --hidden_size=3 --model_name=opfnsvit --manifold=sphere --alpha=1.5 --epochs=1\
 --lr_scheduler_type=binary --max_lr=1e-4\
 --weight_decay=0 --n_layers=1 --n_attn_heads=1 --model_root=.droot/debug-mode --train_bs=64

python -i main.py --model_name=fnsvit --manifold=sphere --alpha=1.5 --a=0\
 --epochs=2 --weight_decay=0 --n_layers=1 --n_attn_heads=1 --model_root=.droot/debug-mode
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='vit-pytorch/main.py training arguments')   
    # training settings 
    #parser.add_argument('--train_with_ddp', default=True, type=bool, help='to use DDP or not')
    parser.add_argument('--max_iters', default=10, type=int)
    #parser.add_argument('--sub', default=1.0, type=float)
    parser.add_argument('--grad_clip', default=0, type=float)
    parser.add_argument('--decay_lr', default=True, type=bool)
    parser.add_argument('--warmup_iters', default=0, type=int)
    #parser.add_argument('--grad_accum_step', default=8, type=int)

    parser.add_argument('--max_lr', default=6e-4, type=float, help='max learning rate')
    parser.add_argument('--min_lr', default=6e-5, type=float, help='min learning rate')
    parser.add_argument('--train_bs', default=2, type=int)
    #parser.add_argument('--eval_bs', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)       

    parser.add_argument('--lr_scheduler_type', default='cosine', type=str, help='cosine | binary') 
    
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
    parser.add_argument("--init_from", default='scratch', type=str, help='scratch | resume | gpt2')
    parser.add_argument('--wandb_log', default=False, type=bool)
    parser.add_argument("--wandb_project", default='image-task', type=str)    

    parser.add_argument('--instance', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)    
    # parser.add_argument('--debug', default=False, type=bool)  # for debuggin
    # parser.add_argument('--lr_scheduler_type', default='constant', type=str)
    # parser.add_argument('--do_train', default=True, type=bool)
    # parser.add_argument('--do_eval', default=True, type=bool)

    # Model settings
    parser.add_argument('--model_name', default='dpvit', type=str)    
    # fnsvit type
    parser.add_argument('--manifold', default='sphere', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)  
    #parser.add_argument('--a', default=1, type=float)
    parser.add_argument('--a', default=0, type=float)
    # sinkvit type
    parser.add_argument('--n_it', default=1, type=int)

    # Dataset settings
    parser.add_argument('--dataset_name', default='cifar10', type=str)
    parser.add_argument('--cache_dir', default=njoin(DROOT, 'cache_dir'), type=str)  

    # Config settings
    parser.add_argument('--qk_share', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--hidden_size', default=48, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_attn_heads', default=2, type=int)
    #parser.add_argument('--intermediate_size', default=4 * 48, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.0, type=float)
    parser.add_argument('--attention_probs_dropout_prob', default=0.0, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)
    # parser.add_argument('--image_size', default=32, type=int)
    # parser.add_argument('--n_classes', default=10, type=int)
    # parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--qkv_bias', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_faster_attn', default=True, type=bool)    


    args = parser.parse_args()    

    # assertions
    model_name = args.model_name.lower()
    if 'fns' in model_name:
        assert args.manifold in ['sphere', 'rd'], 'FNS manifold: sphere or rd'   
        assert 1 <= args.alpha <= 2, 'FNS alpha must be between [1,2]'
        assert args.a in [0,0.5,1], 'Normalization index must be 0 or 0.5 or 1'   

    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    #out_dir = njoin(DROOT, 'ddp_test_stage')      

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
    dataset_name = args.dataset_name
    #gradient_accumulation_steps = args.grad_accum_step # used to simulate larger batch sizes
    batch_size = args.train_bs # if gradient_accumulation_steps > 1, this is the micro-batch size
    #block_size = 1024  # max sequence length (https://stackoverflow.com/questions/66294076/how-to-determine-the-block-size-in-training-a-dataset)

    # model
    n_layer = args.n_layers; n_head = args.n_attn_heads; n_embd = args.hidden_size    
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?

    config = {
        "patch_size": args.patch_size,  # Input image size: 32x32 -> 8x8 patches
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.n_layers,
        "num_attention_heads": args.n_attn_heads,
        "intermediate_size": 4 * args.hidden_size, # 4 * hidden_size
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "initializer_range": args.initializer_range,
        # "image_size": args.image_size,
        # "num_classes": args.n_classes, # num_classes of CIFAR10
        # "num_channels": args.n_channels,
        "qkv_bias": args.qkv_bias,
        "qk_share": args.qk_share,
        "use_faster_attention": args.use_faster_attn,
    }

    attn_setup = {'qk_share': args.qk_share, 'qkv_bias': args.qkv_bias, 'instance': args.instance}    
    attn_setup['dataset_name'] = args.dataset_name    
    if 'fns' in model_name:
        attn_setup['manifold'] = args.manifold
        config['alpha'] = attn_setup['alpha'] = args.alpha      
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth          
        if args.manifold == 'sphere':

            if args.alpha < 2:
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden_size//args.n_attn_heads - 1
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
                config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden_size//args.n_attn_heads  # head_dim                

            model_name = 'rd' + model_name

        # degree index
        config['a'] = attn_setup['a'] = args.a      

    elif 'sinkvit' in model_name:
        config['n_it'] = attn_setup['n_it'] = args.n_it
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth

    attn_setup['model_name'] = model_name        

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
    #device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    #device = f'cuda:{torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if 'cuda' in device else platform.processor()
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    #compile = True # use PyTorch 2.0 to compile the model to be faster
    compile = False
    # -----------------------------------------------------------------------------

    # DDP settings

    # backend = 'nccl' if device == 'cuda' else 'gloo' 
    # # various inits, derived attributes, I/O setup
    # ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?    
    # if ddp and args.train_with_ddp:
    #     ddp_rank = int(os.environ['RANK'])
    #     ddp_local_rank = int(os.environ['LOCAL_RANK'])
    #     ddp_world_size = int(os.environ['WORLD_SIZE'])
    #     #init_process_group(backend=backend)
    #     init_process_group(backend=backend, world_size=ddp_world_size, rank=ddp_rank)  # rank=ddp_local_rank        
    #     if device == 'cuda':
    #         #device = f'cuda:{ddp_local_rank}'
    #         device = f'cuda:{ddp_rank}'
    #         torch.cuda.set_device(device)
    #     else:
    #         device = f'cpu'   
    #         #torch.cpu.set_device(device)     
    #     master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    #     seed_offset = ddp_rank # each process gets a different seed
    #     # world_size number of processes will be training simultaneously, so we can scale
    #     # down the desired gradient accumulation iterations per process proportionally
    #     assert gradient_accumulation_steps % ddp_world_size == 0
    #     gradient_accumulation_steps //= ddp_world_size  
        
    # else:

    # if not ddp, we are running on a single gpu, and one process
    #seed_offset = 0
    ddp_world_size = 1           

    # tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    # print(f"tokens per iteration will be: {tokens_per_iter:,}")
    #images_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
    images_per_iter = ddp_world_size * batch_size
    print(f"images per iteration will be: {images_per_iter:,}")    
    
    if args.model_root == '':
        model_root = structural_model_root(qk_share=args.qk_share, n_layers=args.n_layers,
                                           n_attn_heads=args.n_attn_heads, hidden_size=args.hidden_size  # lr=args.lr, bs=args.train_bs,                                                                                          
                                           )       
        model_root = njoin(DROOT, model_root)
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

    print('-'*25)
    # print(f'ddp = {ddp}')
    # if ddp and args.train_with_ddp:        
    #     print(f'ddp_rank = {ddp_rank}')
    #     print(f'ddp_local_rank = {ddp_local_rank}')
    # print(f'ddp_world_size = {ddp_world_size}')
    print(f'device = {device}')
    #print(f'backend = {backend}')
    print('-'*25 + '\n')           

    #torch.manual_seed(1337 + seed_offset)
    torch.manual_seed(1337 + args.seed)
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

    # poor man's data loader
    if dataset_name == 'cifar10':
        #data_dir = njoin(DROOT,'DATA','cifar-10-batches-py')
        trainloader, testloader, _ = prepare_data(batch_size=batch_size, num_workers=ddp_world_size)    

        if trainloader.dataset[0][0].ndim == 2:
            config['image_size'] = trainloader.dataset[0][0].shape[0]
            config['num_channels'] = 1
        else:
            config['image_size'] = trainloader.dataset[0][0].shape[1]
            config['num_channels'] = trainloader.dataset[0][0].shape[0]

        config['num_classes'] = len(trainloader.dataset.classes)        

    elif dataset_name in ['pathfinder-classification', 'pathx-classification']:  # 'lra-cifar-classification'
        from lra_dataloading import Datasets

        # Get dataset creation function
        create_dataset_fn = Datasets[dataset_name]
        
        # Dataset dependent logic
        if dataset_name in ["imdb-classification", "listops-classification", "aan-classification"]:
            padded = True
            if dataset_name in ["aan-classification"]:
                # Use retreival model for document matching
                retrieval = True
                print("Using retrieval model for document matching")
            else:
                retrieval = False
        else:
            padded = False
            retrieval = False
            
        # Create dataset...
        dataset_obj, trainloader, valloader, testloader, num_classes, seq_len, in_dim, train_size, vocab_size = \
            create_dataset_fn(cache_dir=args.cache_dir, seed=args.seed, train_bs=batch_size, eval_bs=batch_size)
        eval_size = len(testloader.dataset)          

        if trainloader.dataset[0][0].ndim == 2:
            config['image_size'] = trainloader.dataset[0][0].shape[0]
            config['num_channels'] = 1
        else:
            config['image_size'] = trainloader.dataset[0][0].shape[1]
            config['num_channels'] = trainloader.dataset[0][0].shape[0]

        config['num_classes'] = trainloader.dataset.tensors[1].unique().shape[0]
        
    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']
    assert config['image_size'] % config['patch_size'] == 0

    train_size = len(trainloader.dataset)
    eval_size = len(testloader.dataset)                      
    steps_per_epoch = len(trainloader)

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

    def get_batch(split):
        if split == 'train':
            data = trainloader
        else:
            data = testloader
        ix = torch.randint(len(data), (batch_size,))
        x = torch.stack([data.dataset[i][0] for i in ix])
        y = torch.tensor([data.dataset[i][1] for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y        
        #return x, y, ix

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    # meta_path = os.path.join(data_dir, 'meta.pkl')
    # meta_vocab_size = None
    # if os.path.exists(meta_path):
    #     with open(meta_path, 'rb') as f:
    #         meta = pickle.load(f)
    #     meta_vocab_size = meta['vocab_size']
    #     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init FOR GPT2
    # model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    #                   bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print(f'Initializing a new {model_name} from scratch \n')

        if model_name == 'dpvit':
            from vit_pytorch.vit import ViTForClassfication
            model = ViTForClassfication(config)
        if model_name == 'opdpvit':
            from vit_pytorch.opvit import OPViTForClassfication
            model = OPViTForClassfication(config)            
        # elif model_name == 'fnsvit':
        #     from vit_pytorch.fns_vit import FNSViTForClassfication
        #     model = FNSViTForClassfication(config)    
        # elif model_name == 'opfnsvit':
        #     from vit_pytorch.opfns_vit import OPFNSViTForClassfication
        #     model = OPFNSViTForClassfication(config)     
        elif model_name == 'spfnsvit':
            from vit_pytorch.spfns_vit import SPFNSViTForClassfication
            model = SPFNSViTForClassfication(config)    
        elif model_name == 'spopfnsvit':
            from vit_pytorch.spopfns_vit import SPOPFNSViTForClassfication
            model = SPOPFNSViTForClassfication(config)      
        elif model_name == 'rdfnsvit':
            from vit_pytorch.rdfns_vit import RDFNSViTForClassfication
            model = RDFNSViTForClassfication(config)    
        elif model_name == 'rdopfnsvit':
            from vit_pytorch.rdopfns_vit import RDOPFNSViTForClassfication
            model = RDOPFNSViTForClassfication(config)               
        elif model_name == 'sinkvit':
            from vit_pytorch.sink_vit import SINKViTForClassfication
            model = SINKViTForClassfication(config)      
        elif model_name == 'opsinkvit':
            from vit_pytorch.opsink_vit import OPSINKViTForClassfication
            model = OPSINKViTForClassfication(config)                         

    """
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    """

    # crop down the model block size if desired, using model surgery
    # if block_size < model.config.block_size:
    #     model.crop_block_size(block_size)
    #     model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # ++++++++++ SOMEHOW THIS WORKS FOR CPUS AS WELL ++++++++++
    # initialize a GradScaler. If enabled=False scaler is a no-op
    # if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # else:
    #     scaler = torch.GradScaler('cpu', enabled=(dtype == 'float16'))

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    #optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)    
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1,beta2), weight_decay=args.weight_decay)

    # if init_from == 'resume':
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        #model = torch.compile(model) # requires PyTorch 2.0
        model = torch.compile(model, backend='aot_eager')

    # wrap model into DDP container
    # if ddp:
    #     if device_type == 'cuda':
    #         model = DDP(model, device_ids=[ddp_local_rank])
    #     else:
    #         model = DDP(model, device_ids=[], output_device=[])
    #     #model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)    

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
        for batch in testloader:    

            if args.dataset_name == 'cifar10':
                data, label = batch
            else:
                data, label, _ = batch
            if data.ndim == 3:
                data = data[:,None]
            data = data.to(device)
            label = label.to(device)

            val_logits, _ = model(data)
            val_loss = loss_fn(val_logits, label)

            acc = (val_logits.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(testloader)
            epoch_val_loss += val_loss / len(testloader)  

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
    
    # logging
    if wandb_log:
        import wandb  # NOT IN CONTAINER
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    #X, Y, IX = get_batch('train')
    #IXs = IX
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    #raw_model = model.module if ddp else model # unwrap DDP container if needed
    raw_model = model
    running_mfu = -1.0

    if not wandb_log:
        metrics_ls = []    

    t0 = time.time()
    dt = None
    iter_num = 0
    metric_cols = ['iter', 'lr', 'train_loss', 'val_loss', 'train_acc', 'val_acc','secs_per_eval']
    #for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        print('epoch: ', epoch)
        epoch_loss = 0
        epoch_accuracy = 0

        #for data, label in trainloader:
        for batch in tqdm(trainloader):

            if args.dataset_name == 'cifar10':
                data, label = batch
            else:
                data, label, _ = batch
            if data.ndim == 3:
                data = data[:,None]                
            data = data.to(device)
            label = label.to(device)

            #output = model(data)
            logits, _ = model(data)
            loss = loss_fn(logits, label)

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

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            loss.backward()
            # clip the gradient
            if grad_clip != 0.0:                
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            acc = (logits.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)

            # if iter_num == 0 and eval_only:
            #     break

            iter_num += 1

        epoch_val_accuracy, epoch_val_loss = fb_estimate_val()

        # evaluate the loss on train/val sets and write checkpoints            
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": epoch_loss,
                "val/loss": epoch_val_loss,
                "train/acc": epoch_accuracy,
                "val/acc": epoch_val_accuracy,
                "lr": lr,
                "dt": dt
                #"mfu": running_mfu*100, # convert to percentage
            })
        else:
            metrics_ls.append([iter_num, lr, epoch_loss.item(), epoch_val_loss.item(), epoch_accuracy.item(), epoch_val_accuracy.item(), dt])

            df = pd.DataFrame(metrics_ls, columns=metric_cols)
            df.to_csv(njoin(out_dir, '_run_performance.csv'))               

        if epoch_val_loss < best_val_loss or always_save_checkpoint:
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

    if not wandb_log:
        if isfile(njoin(out_dir, '_run_performance.csv')):
            os.remove(njoin(out_dir, '_run_performance.csv'))
        df = pd.DataFrame(metrics_ls, columns=metric_cols)
        df.to_csv(njoin(out_dir, 'run_performance.csv'))

    print(f'All data saved under {out_dir}')        