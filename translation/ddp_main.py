import argparse
import json
import os
import pandas as pd
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from contextlib import nullcontext
from os.path import isdir, isfile
from os import makedirs

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#from model import GPTConfig, GPT

from constants import *
from mutils import njoin, create_model_dir, convert_train_history, structural_model_root
from data_utils import prepare_data, BilingualDataset

from torch.optim import AdamW

from transformers import AutoTokenizer

# import torchmetrics

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

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
python -i ddp_main.py --model_name=fnstranslation --beta=1.5\
 --max_iters=100 --eval_interval=5 --eval_iters=200 --weight_decay=0 --model_root=.droot/single-core 
"""

# multi-core
"""
torchrun --nnodes=1 --nproc_per_node=4 ddp_main.py --model_name=fnsvit --beta=1.5 --max_iters=100 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2 --model_root=.droot/multi-core
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='translation/main.py training arguments')   
    # training settings 
    parser.add_argument('--train_with_ddp', default=True, type=bool, help='to use DDP or not')
    parser.add_argument('--max_iters', default=10, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--decay_lr', default=True, type=bool)
    parser.add_argument('--warmup_iters', default=0, type=int)
    parser.add_argument('--grad_accum_step', default=8, type=int)

    parser.add_argument('--max_lr', default=6e-4, type=float, help='max learning rate')
    parser.add_argument('--min_lr', default=6e-5, type=float, help='min learning rate')
    parser.add_argument('--train_bs', default=2, type=int)
    parser.add_argument('--eval_bs', default=10, type=int)
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
    
    parser.add_argument("--save_model_every", default=0, type=int)
    parser.add_argument("--exp_name", default='translation-task', type=str)
    parser.add_argument("--init_from", default='scratch', type=str, help='scratch | resume | gpt2')
    parser.add_argument('--wandb_log', default=False, type=bool)
    parser.add_argument("--wandb_project", default='translation-task', type=str)    

    # parser.add_argument('--seed', default=42, type=int)    
    # parser.add_argument('--debug', default=False, type=bool)  # for debuggin
    # parser.add_argument('--lr_scheduler_type', default='constant', type=str)
    # parser.add_argument('--do_train', default=True, type=bool)
    # parser.add_argument('--do_eval', default=True, type=bool)

    # Model settings
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--qk_share', default=False, type=bool)
    parser.add_argument('--model_name', default='dptranslation', type=str)    
    parser.add_argument('--bandwidth', default=1, type=float)  
    parser.add_argument('--sphere_radius', default=1, type=float)  

    # Dataset settings
    parser.add_argument('--dataset_name', default='iwslt14', type=str)
    parser.add_argument('--divider', default=1, type=int)  # downsizing the test dataset    
    parser.add_argument('--tokenizer_path', default=None, type=str)  # tokenizer file path

    # Config settings
    parser.add_argument('--hidden_size', default=48, type=int)
    # parser.add_argument('--intermediate_size', default=4 * 48, type=int)    
    parser.add_argument('--num_encoder_layers', default=1, type=int)
    parser.add_argument('--num_decoder_layers', default=1, type=int)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.0, type=float)
    parser.add_argument('--encoder_dropout_prob', default=0.0, type=float)
    parser.add_argument('--decoder_dropout_prob', default=0.0, type=float)
    parser.add_argument('--attention_probs_dropout_prob', default=0.0, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)
    parser.add_argument('--qkv_bias', default=True, type=bool)
    parser.add_argument('--use_faster_attn', default=True, type=bool)
    parser.add_argument('--src_vocab_size', default=1000, type=int)  
    parser.add_argument('--src_pad_token_id', default=0, type=int)  
    parser.add_argument('--trg_vocab_size', default=1000, type=int)  
    parser.add_argument('--trg_pad_token_id', default=0, type=int)  
    parser.add_argument('--max_length', default=128, type=int)  


    args = parser.parse_args()    

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
    wandb_run_name = f'{args.model_name}-{args.dataset_name}' # 'run' + str(time.time())

    # data
    dataset = args.dataset_name
    gradient_accumulation_steps = args.grad_accum_step # used to simulate larger batch sizes
    batch_size = args.train_bs # if gradient_accumulation_steps > 1, this is the micro-batch size
    #block_size = 1024  # max sequence length (https://stackoverflow.com/questions/66294076/how-to-determine-the-block-size-in-training-a-dataset)

    # model
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    
    config = {
        "beta": args.beta,
        "bandwidth": args.bandwidth,
        "sphere_radius": args.sphere_radius,
        "hidden_size": args.hidden_size,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": 4 * args.hidden_size, # 4 * hidden_size
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "encoder_dropout_prob": args.encoder_dropout_prob,
        "decoder_dropout_prob": args.decoder_dropout_prob,
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "initializer_range": args.initializer_range,
        "qkv_bias": args.qkv_bias,
        "use_faster_attention": args.use_faster_attn,
        "src_vocab_size": args.src_vocab_size,
        "src_pad_token_id": args.src_pad_token_id,
        "trg_vocab_size": args.trg_vocab_size,
        "trg_pad_token_id": args.trg_pad_token_id,
        "max_length": args.max_length,
        "tokenizer_path": args.tokenizer_path,
    }
    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']

    attn_setup = {'qk_share': False}
    attn_setup['model_name'] = args.model_name
    attn_setup['dataset_name'] = args.dataset_name    
    if args.model_name == 'fnstranslation':
        config['beta'] = attn_setup['beta'] = args.beta      
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth   
        if args.beta < 2:            
            config['d_intrinsic'] = int(config["hidden_size"] / config["num_attention_heads"])  # head_dim
            config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic'])-1)/np.pi)                            
        else:
            config['sphere_radius'] = 1

        attn_setup['sphere_radius'] = config['sphere_radius']       
        attn_setup['mask_val'] = np.pi * config['sphere_radius']      

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
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    #compile = True # use PyTorch 2.0 to compile the model to be faster
    compile = False
    # -----------------------------------------------------------------------------
    # CONFIGS FOR GPT2
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # DDP settings
    backend = 'nccl' if device == 'cuda' else 'gloo' 

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?    
    if ddp and args.train_with_ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        #init_process_group(backend=backend)
        init_process_group(backend=backend, world_size=ddp_world_size, rank=ddp_rank)  # rank=ddp_local_rank        
        if device == 'cuda':
            #device = f'cuda:{ddp_local_rank}'
            device = f'cuda:{ddp_rank}'
            torch.cuda.set_device(device)
        else:
            device = f'cpu'   
            #torch.cpu.set_device(device)     
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size  
        
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process:
        # save config
        with open("config.json", "w") as ofile: 
            json.dump(config, ofile)   
        # save attn_setup
        with open("attn_setup.json", "w") as ofile: 
            json.dump(attn_setup, ofile)                 

        print('-'*25)
        print(f'ddp = {ddp}')
        if ddp and args.train_with_ddp:        
            print(f'ddp_rank = {ddp_rank}')
            print(f'ddp_local_rank = {ddp_local_rank}')
        print(f'ddp_world_size = {ddp_world_size}')
        print(f'device = {device}')
        print(f'backend = {backend}')
        print('-'*25 + '\n')              
    
    if args.model_root == '':
        model_root = structural_model_root(qk_share=args.qk_share, num_encoder_layers=args.num_encoder_layers,
                                           num_decoder_layers=args.num_decoder_layers, num_attention_heads=args.num_attention_heads,
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

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
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

    def get_all_sentences(ds, lang):
        for item in ds:
            yield item['translation'][lang]
            
    def get_or_build_tokenizer(config, ds, lang):
        if config['tokenizer_path'] is None:
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
            # tokenizer.save(str(tokenizer_path))
        else:
            tokenizer_path = Path(config['tokenizer_path'].format(lang))
            assert Path.exists(tokenizer_path), "Tokenizer path does not exist"
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
    
    # def preprocess_function(examples, tokenizer, language, max_length):
    #     inputs = [example[language] for example in examples["translation"]]
    #     pad_id = tokenizer.token_to_id("[PAD]")
    #     encodings = tokenizer.encode_batch(inputs)
    #     for encoding in encodings:
    #         encoding.truncate(max_length=max_length)
    #         encoding.pad(length=max_length, pad_id=pad_id, pad_token="[PAD]")
    #     return [encoding.ids for encoding in encodings]
    
    # def preprocess_function(examples, tokenizer_src, src_language, tokenizer_trg, trg_language, max_length):
    #     inputs = [example[src_language] for example in examples["translation"]]
    #     targets = [example[trg_language] for example in examples["translation"]]
        
    #     model_inputs = tokenizer_src(inputs, text_target=targets, max_length=128, truncation=True)
    #     return model_inputs
        
        
    #     pad_id = tokenizer_src.token_to_id("[PAD]")
    #     model_inputs = tokenizer_src.encode_batch(inputs)
    #     for model_input in model_inputs:
    #         model_input.truncate(max_length=max_length)
    #         model_input.pad(length=max_length, pad_id=pad_id, pad_token="[PAD]")
    #     model_inputs = [model_input.ids for model_input in model_inputs]
    #     labels = tokenizer_trg.encode_batch(targets)
    #     for label in labels:
    #         label.truncate(max_length=max_length)
    #         label.pad(length=max_length, pad_id=pad_id, pad_token="[PAD]")
    #     labels = [label.ids for label in labels]
    #     return model_inputs, labels

    if dataset == 'iwslt14': 
        src_language, trg_language = 'de', 'en'
        dataset = load_dataset("ted_talks_iwslt", language_pair=(src_language, trg_language), year="2014", split="train")
        tokenizer_src = get_or_build_tokenizer(config, dataset, src_language)
        tokenizer_trg = get_or_build_tokenizer(config, dataset, trg_language)
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.2, shuffle=True) #20% test set
        #trainset, testset = dataset
        trainset = dataset['train']; testset = dataset['test']
        trainset = BilingualDataset(trainset, tokenizer_src, tokenizer_trg, src_language, trg_language, config["max_length"])
        testset = BilingualDataset(testset, tokenizer_src, tokenizer_trg, src_language, trg_language, config["max_length"])
        # Create dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=ddp_world_size)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=ddp_world_size)
        # Get pad token ids and vocab sizes
        config["src_vocab_size"] = tokenizer_src.get_vocab_size()
        config["src_pad_token_id"] = tokenizer_src.token_to_id("[PAD]")
        config["trg_vocab_size"] = tokenizer_trg.get_vocab_size()
        config["trg_pad_token_id"] = tokenizer_trg.token_to_id("[PAD]")

    # only for large datasets
    # def get_batch(split):
    #     # We recreate np.memmap every batch to avoid a memory leak, as per
    #     # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    #     if split == 'train':
    #         data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    #     else:
    #         data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    #     ix = torch.randint(len(data) - block_size, (batch_size,))
    #     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    #     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    #     if device_type == 'cuda':
    #         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    #     else:
    #         x, y = x.to(device), y.to(device)
    #     return x, y

    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    def get_batch(split):
        if split == 'train':
            data = trainloader
        else:
            data = testloader
        ix = torch.randint(len(data), (batch_size,))
        # x = torch.tensor([data.dataset["encoder_input"][i] for i in ix]) # (B, N) 
        # y = torch.tensor([data.dataset["decoder_input"][i] for i in ix]) # (B, N) 
        # encoder_mask = torch.tensor([data.dataset["attention_mask"][i] for i in ix]) # (B, N) 
        # encoder_mask = (encoder_mask.unsqueeze(-1)@encoder_mask.unsqueeze(1)).view(batch_size, 1, config["max_length"], config["max_length"]) # (B,1,N,N)
        # decoder_mask = torch.stack([(torch.tensor(data.dataset["attention_mask"][i] != 0)).unsqueeze(0).int() & causal_mask(config["max_length"]) for i in ix]) # (B,1,N,N)

        ##### CHAGNES HERE #####
        # this seems slow
        data_points = [data.dataset.__getitem__(i.item()) for i in ix]        
        x = torch.stack([data_points[i]['encoder_input'] for i in range(batch_size)])
        y = torch.stack([data_points[i]['decoder_input'] for i in range(batch_size)])
        pre_encoder_mask = torch.stack([data_points[i]['encoder_mask'].squeeze() for i in range(batch_size)])
        encoder_mask = (pre_encoder_mask.unsqueeze(-1)@pre_encoder_mask.unsqueeze(1)).view(batch_size, 1, config["max_length"], config["max_length"]) # (B,1,N,N)
        decoder_mask = torch.stack([data_points[i]['decoder_mask'] for i in range(batch_size)]) # (B,1,N,N)

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            encoder_mask, decoder_mask = encoder_mask.pin_memory().to(device, non_blocking=True), decoder_mask.pin_memory().to(device, non_blocking=True)
        else:
            x, y, encoder_mask, decoder_mask = x.to(device), y.to(device), encoder_mask.to(device), decoder_mask.to(device)
        return x, y, encoder_mask, decoder_mask    

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
        print(f'Initializing a new {args.model_name} from scratch \n')
        # determine the vocab size we'll use for from-scratch training
        # if meta_vocab_size is None:
        #     print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        # model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        # gptconf = GPTConfig(**model_args)
        # model = GPT(gptconf)
        if args.model_name == 'dptranslation':
            from models.translation import DPForTranslation
            model = DPForTranslation(config)
        elif args.model_name == 'fnstranslation':
            from models.fns_translation import FNSForTranslation
            model = FNSForTranslation(config)    
        else:
            print(f'{args.model_name} does not exist!')
            quit()

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

    ##### CHANGES HERE #####
    # loss function    
    #loss_fn = nn.CrossEntropyLoss(ignore_index=config["src_pad_token_id"])
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
    if ddp:
        if device_type == 'cuda':
            model = DDP(model, device_ids=[ddp_local_rank])
        else:
            model = DDP(model, device_ids=[], output_device=[])
        #model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)    

    # helps estimate an arbitrarily accurate loss over either split using many batches
    def greedy_decode(model, source, source_mask, tokenizer_trg, max_len, device):        
        # sos_idx = tokenizer_trg.bos_token_id
        # eos_idx = tokenizer_trg.eos_token_id
        ##### CHANGES HERE #####
        sos_idx = tokenizer_trg.get_vocab()['[SOS]']
        eos_idx = tokenizer_trg.get_vocab()['[EOS]']

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encoder(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build causal mask for target 
            decoder_mask = torch.stack([(source_mask[i,0,0,:] != 0).unsqueeze(0).int() & causal_mask(config["max_length"]) for i in range(source_mask.shape[0])]) # (B,1,N,N)
            decoder_mask = decoder_mask.type_as(source_mask).to(device)

            ##### CHANGES HERE #####
            # calculate output
            out = model.decoder(decoder_input, encoder_output, source_mask, decoder_mask)
            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    @torch.no_grad()
    def estimate_loss():
        out_loss = {}
        out_bleu = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            predicted = []
            expected = []
            for k in range(eval_iters):
                X, Y, encoder_mask, decoder_mask = get_batch(split)
                with ctx:
                    ##### CHANGES HERE #####
                    logits, _, _, _ = model(X, encoder_mask, decoder_mask)
                    loss = loss_fn(logits, F.one_hot(Y, num_classes=config['trg_vocab_size']).float())                

                    # Generate sentence
                    model_out = greedy_decode(model, X, encoder_mask, tokenizer_trg, config["max_length"], device)
                    model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())
                    predicted.append(model_out_text)
                    expected.append(tokenizer_trg.decode(Y.detach().cpu().numpy()))
                losses[k] = loss.item()
            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            out_loss[split] = losses.mean()
            out_bleu[split] = bleu
        model.train()
        return out_loss, out_bleu

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
    if wandb_log and master_process:
        import wandb  # NOT IN CONTAINER
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
    # training loop
    X, Y, encoder_mask, decoder_mask = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0

    if not wandb_log:
        metrics_ls = []    
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses, bleu = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train bleu {bleu['train']:.4f}, val bleu {bleu['val']:.4f}")
            if wandb_log: # CHANGE
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "train/bleu": bleu['train'],
                    "val/bleu": bleu['val'],
                    "lr": lr
                    #"mfu": running_mfu*100, # convert to percentage
                })
            else:
                metrics_ls.append([iter_num, lr, losses['train'].item(), losses['val'].item(), bleu['train'].item(), bleu['val'].item()])

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
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
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                ##### CHANGES HERE #####
                logits, _, _, _ = model(X, encoder_mask, decoder_mask)
                # logits (bs, seq_len, tgt_vocab_size)
                loss = loss_fn(logits, torch.F.one_hot(Y, num_classes=config['trg_vocab_size']).float())  # logits.argmax(-1)                
                # loss = loss_fn(logits.view(-1, config["trg_vocab_size"]), Y.view(-1)) 
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, encoder_mask, decoder_mask = get_batch('train')
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

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            # if local_iter_num >= 5: # let the training loop settle a bit
            #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)  # mfu (model flop utilization)
            #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            #print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            #if master_process:
            if not wandb_log:
                df = pd.DataFrame(metrics_ls, columns=['iter', 'lr', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
                df.to_csv(njoin(out_dir, 'run_performance.csv'))
            break

    if ddp:
        destroy_process_group()
        
        
# Tokenizer - which tokenizers, import tokenizers, preprocess data in datautils
# Train - WHat is the task? How does it know when to finish generating? 
# BLEU - correct implementation of BLEU score? 