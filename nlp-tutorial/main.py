import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
from torch.utils.data import DataLoader

from models.model_utils import create_longformer_mask
from utils.data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
#from trainer import Trainer

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

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

from constants import *
from utils.mutils import njoin, create_model_dir, convert_train_history, structural_model_root
from utils.mutils import str2bool, str2ls, str_or_float

#from torch.optim import AdamW
from torch.optim import Adam

"""
s = 'model_name=fnsformer,alpha=1.2,a=0,bandwidth=1,manifold=v2_rd,is_op=True,dataset=emotion,model_root=/project/RDS-FSC-frac_attn-RW/fractional-attn/nlp-tutorial/.droot/1L-hidden=32-max_len=512-rescaled/config_qqv/emotion/layers=1-heads=1-qqv,seed=0,qk_share=True,is_rescale_dist=True,weight_decay=0,max_lr=0.001,min_lr=0.0002,max_len=512,n_layers=1,n_attn_heads=1,epochs=25,fix_embed=False,lr_scheduler_type=binary,train_bs=16,hidden=32'
ss = '--' + ' --'.join(s.split(','))

##### trained-embed #####
python -i main.py --model_name=fnsformer --alpha=1 --a=0 --bandwidth=1 --manifold=rd --is_op=True\
 --dataset=imdb --model_root=/project/RDS-FSC-frac_attn-RW/fractional-attn/nlp-tutorial/.droot/debug-mode --seed=0\
 --qk_share=False --is_rescale_dist=True --weight_decay=0 --max_lr=0.001 --min_lr=0.0002\
 --max_len=512 --n_layers=1 --n_attn_heads=1 --epochs=25 --fix_embed=False --lr_scheduler_type=binary --train_bs=16 --hidden=32
"""

"""
python -i main.py --train_bs=32 --dataset_name=imdb --fix_embed=True --pretrained_model_name=glove\
 --hidden=300 --model_name=dpformer --qk_share=False\
 --epochs=1 --weight_decay=0 --n_layers=1 --n_attn_heads=1 --model_root=.droot/debug-mode
"""

"""
python -i main.py --train_bs=32 --dataset_name=imdb --fix_embed=True --pretrained_model_name=glove --max_len=1024\
 --hidden=16 --model_name=fnsformer --manifold=rd --is_op=True --alpha=1.2 --bandwidth=1 --a=0  --qk_share=True\
 --max_lr=0.0001 --lr_scheduler_type=constant --epochs=5 --weight_decay=0\
 --n_layers=1 --n_attn_heads=1 --seed=0\
 --model_root=.droot/debug-mode-v3
"""

"""
python -i main.py --train_bs=32 --dataset_name=imdb --fix_embed=True --hidden=16\
 --model_name=fnsformer --manifold=rd --alpha=1.2 --bandwidth=1 --qk_share=False\
 --epochs=1 --weight_decay=0 --n_layers=1 --n_attn_heads=1 --model_root=.droot/debug-mode
"""

# single-core
"""
python -i main.py --train_bs=32 --dataset_name=imdb\
  --model_name=fnsformer --manifold=sphere --alpha=1.2 --bandwidth=1 --qk_share=False\
 --epochs=1 --weight_decay=0 --n_layers=1 --n_attn_heads=1 --model_root=.droot/debug-mode 
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
    parser.add_argument('--train_mask_type', default=None, type=str)

    #parser.add_argument('--eval_bs', default=10, type=int)
    #parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--beta1', default=0.9, type=float) 
    parser.add_argument('--beta2', default=0.999, type=float)  # 0.95       

    parser.add_argument('--lr_scheduler_type', default='constant', type=str, help='cosine | binary | constant') 
    
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

    parser.add_argument('--seed', default=0, type=int)    

    # Tokenizer
    parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--tokenizer_name', default='sentencepiece', type=str)  
    parser.add_argument('--pretrained_model',    default='wiki.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')

    # Dataset settings
    parser.add_argument('--dataset_name', default='imdb', type=str)
    parser.add_argument('--cache_dir', default=njoin(DROOT, 'cache_dir'), type=str)  

    # Model settings
    parser.add_argument('--model_name', default='spfnsvit', type=str)  
    parser.add_argument('--is_resnet_scale', type=str2bool, nargs='?', const=True, default=False)
    # fns type
    parser.add_argument('--manifold', default='sphere', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--bandwidth', default=1)  # type=float  
    parser.add_argument('--a', default=0, type=float)
    parser.add_argument('--is_rescale_dist', type=str2bool, nargs='?', const=True, default=False) 
    # sink type
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

    parser.add_argument('--fix_embed', type=str2bool, nargs='?', const=True, default=False) 
    parser.add_argument('--pretrained_model_name', default='glove', type=str, help='distilbert-base-uncased | albert-base-v2 | glove')

    args = parser.parse_args()    

    # assertions
    model_name = args.model_name.lower()
    if 'fns' in model_name:
        assert args.manifold in ['sphere', 'rd', 'v2_rd'], 'FNS manifold: sphere or rd'   
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
    #device = 'cpu'  # for debugging
    device_name = torch.cuda.get_device_name(0) if 'cuda' in device else platform.processor()
    # -----------------------------------------------------------------------------        

    print('-'*25)
    print(f'device = {device}')
    print('-'*25 + '\n')           

    ##### SET SEED #####
    torch.manual_seed(args.seed)   
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        
    args.bandwidth = float(args.bandwidth) if args.bandwidth.replace('.','').isnumeric() else args.bandwidth
    # poor man's data loader
    if dataset_name == 'imdb':
        if not args.fix_embed:
            # Load tokenizer
            if args.tokenizer_name == 'sentencepiece':
                tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
            else:
                tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
                tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file =args.vocab_file)      

        else:
            if args.pretrained_model_name == 'distilbert-base-uncased':
                from transformers import AutoTokenizer, DistilBertModel
                #tokenizer = AutoTokenizer.from_pretrained(f'distilbert/{args.pretrained_model_name}')
                tokenizer = AutoTokenizer.from_pretrained(f'distilbert/distilbert-base-uncased')
                pretrained_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
                vocab_size, pretrained_model_hidden =\
                     pretrained_model.embeddings.word_embeddings.weight.shape
                pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

                assert args.max_len == pretrained_seq_len - 1, f'args.max_len does not match {pretrained_seq_len}!'

            elif args.pretrained_model_name == 'albert-base-v2':
                from transformers import AlbertTokenizer, AlbertModel
                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                pretrained_model = AlbertModel.from_pretrained("albert-base-v2")                
                vocab_size, pretrained_model_hidden =\
                     pretrained_model.embeddings.word_embeddings.weight.shape
                pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

                assert args.max_len == pretrained_seq_len - 1, f'args.max_len does not match {pretrained_seq_len}!'

            elif args.pretrained_model_name == 'gpt2':
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2")                                
                vocab_size, pretrained_model_hidden =\
                     pretrained_model.transformer.wte.weight.shape
                pretrained_seq_len, _ = pretrained_model.transformer.wpe.weight.shape

                assert args.max_len == pretrained_seq_len - 1, f'args.max_len does not match {pretrained_seq_len}!'

            elif args.pretrained_model_name == 'glove':

                from torchtext.data.utils import get_tokenizer
                from torchtext.vocab import GloVe
                from constants import GLOVE_DIMS
                
                if args.hidden in GLOVE_DIMS:
                    print(f'Pretrained dimension {args.hidden} deployed! \n')

                    # tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, 
                    #                                 vocab_file=njoin(DROOT, 'GLOVE', f'vocab_npa_d={args.hidden}.npy'))
                    tokenizer = get_tokenizer("basic_english")
                    glove = GloVe(name='6B', dim=args.hidden)
                    glove_dim = args.hidden
                else:
                    assert args.hidden < max(GLOVE_DIMS), 'Maximal glove dim is 300!'
                    print(f'Reduced dimension {args.hidden} deployed! \n')

                    # for glove_dim in GLOVE_DIMS:
                    #     if glove_dim >= args.hidden:
                    #         break
                    glove_dim = 300
                    tokenizer = get_tokenizer("basic_english")                    
                    glove = GloVe(name='6B', dim=glove_dim)
                vocab_size = len(glove.stoi)    

            #max_length = tokenizer.model_max_length - 1
            #max_length = tokenizer.model_max_length - 2
            #args.max_len = tokenizer.model_max_length - 2
            #args.max_len = tokenizer.model_max_length - 1

            # def preprocess_function(examples):
            #     return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

        # Build DataLoader
        if args.fix_embed and args.pretrained_model_name == 'glove':
            from utils.data_utils import glove_create_examples
            train_dataset = glove_create_examples(args, glove_dim, tokenizer, mode='train')
            test_dataset = glove_create_examples(args, glove_dim, tokenizer, mode='test')            
        else:
            train_dataset = create_examples(args, tokenizer, mode='train')
            test_dataset = create_examples(args, tokenizer, mode='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

        train_size = len(train_loader.dataset)
        eval_size = len(test_loader.dataset)                      
        steps_per_epoch = len(train_loader)        
    else:
        assert not args.fix_embed, f'fix_embed cannot be done for dataset {args.dataset_name}'
        from transformers import AutoTokenizer
        from utils.data_utils import get_datasets, datasets_create_examples

        # Load a tokenizer (Example: BERT)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenized_train_dataset, tokenized_test_dataset = get_datasets(args, tokenizer)        
        train_loader, test_loader = datasets_create_examples(args, 
                                                             tokenized_train_dataset, 
                                                             tokenized_test_dataset)        
        train_size = len(tokenized_train_dataset)
        eval_size = len(tokenized_test_dataset)  
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

    # init a new model from scratch
    print(f'Initializing a new {model_name} from scratch \n')

    if not (args.fix_embed and args.pretrained_model_name == 'glove'):
        vocab_size = tokenizer.vocab_size
    if (args.fix_embed and args.pretrained_model_name == 'glove'):
        pad_token_id = 0  
    else: 
        pad_token_id = tokenizer.pad_token_id

    # These are not hard constraints, but are used to prevent misconfigurations
    assert args.hidden % args.n_attn_heads == 0    
    config = {
        "is_resnet_scale": args.is_resnet_scale,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "n_heads": args.n_attn_heads,
        "head_dim": args.hidden//args.n_attn_heads,
        "d_model": args.hidden,
        "d_ff": args.ffn_hidden, 
        "p_drop": args.dropout,  
        "vocab_size": vocab_size, 
        "seq_len": args.max_len,
        "pad_id": pad_token_id,
        "qkv_bias": args.qkv_bias,
        "qk_share": args.qk_share,     
        "is_op":    args.is_op,
        "fix_embed": args.fix_embed,
        "type_vocab_size": None        
    }
    config['device'] = device

    config['train_mask_type'] = train_mask_type = args.train_mask_type
    if train_mask_type is not None:
        assert train_mask_type in ['longformer'], 'train_mask_type does not exist!'     
        if train_mask_type == 'longformer':             
            global_token_indices = [0,1]
            config['train_mask'] = create_longformer_mask(args.max_len, args.max_len//8, 
                                                          global_token_indices).bool().to(device) 

    # dataset
    if args.dataset_name == 'imdb':
        config['num_classes'] = 2
    else:
        config['num_classes'] = len(tokenized_train_dataset['label'].unique())

    attn_setup = {'qk_share': args.qk_share, 'qkv_bias': args.qkv_bias, 'is_op': args.is_op,
                  'fix_embed': args.fix_embed, 
                  'dataset_name': args.dataset_name}        

    if args.fix_embed:
        from models.model_utils import load_pretrained_model, load_embeddings

        # config['layer_norm_eps'] = pretrained_model.embeddings.LayerNorm.eps
        # config['hidden_dropout_prob'] = pretrained_model.embeddings.dropout.p
        config['pretrained_model_name'] = args.pretrained_model_name
        if config['pretrained_model_name'] == 'glove':
            config['sinusoidal_pos_embds'] = True
        elif config['pretrained_model_name'] in ['distilbert-base-uncased', 'albert-base-v2', 'gpt2']:
            config['sinusoidal_pos_embds'] = False

        attn_setup['pretrained_model_name'] = args.pretrained_model_name

        if args.pretrained_model_name in ['distilbert-base-uncased', 'albert-base-v2']:
            #pretrained_model = load_pretrained_model(config)
            if args.hidden == pretrained_model_hidden:
                pretrained_word_embeddings =\
                    pretrained_model.embeddings.word_embeddings.detach().numpy()
                pretrained_position_embeddigs =\
                    pretrained_model.embeddings.position_embeddings.detach().numpy()
            else:           
                # PCA
                from sklearn.decomposition import PCA  
                pca = PCA(n_components=args.hidden, random_state=args.seed) 
                pretrained_word_embeddings =\
                     pca.fit_transform(pretrained_model.embeddings.word_embeddings.weight.detach().numpy())    
                pretrained_position_embeddings =\
                    pca.fit_transform(pretrained_model.embeddings.position_embeddings.weight.detach().numpy())                  
        elif args.pretrained_model_name == 'gpt2':
            if args.hidden == pretrained_model_hidden:
                pretrained_word_embeddings =\
                    pretrained_model.transformer.wte.weight.detach().numpy()
                pretrained_position_embeddigs =\
                    pretrained_model.transformer.wpe.weight.detach().numpy()
            else:           
                # PCA
                from sklearn.decomposition import PCA  
                pca = PCA(n_components=args.hidden, random_state=args.seed) 
                pretrained_word_embeddings =\
                     pca.fit_transform(pretrained_model.transformer.wte.weight.detach().numpy())    
                pretrained_position_embeddings =\
                    pca.fit_transform(pretrained_model.transformer.wte.weight.detach().numpy())             
        elif args.pretrained_model_name == 'glove':      
            if args.hidden in GLOVE_DIMS:                 
                pretrained_word_embeddings = glove.vectors
            else:
                # t-SNE
                # from sklearn.manifold import TSNE  # use t-SNE by default due to nonlinearity
                # tsne = TSNE(n_components=args.hidden, perplexity=30, random_state=args.seed)
                # pretrained_word_embeddings = tsne.fit_transform(glove.vectors.numpy())                
                # PCA
                from sklearn.decomposition import PCA  
                pca = PCA(n_components=args.hidden, random_state=args.seed)
                pretrained_word_embeddings = pca.fit_transform(glove.vectors.numpy())                
                
    if 'fns' in model_name:
        attn_setup['manifold'] = args.manifold
        config['alpha'] = attn_setup['alpha'] = args.alpha      
        config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth          
        if args.manifold == 'sphere':

            model_name = 'sp' + model_name
            config['is_rescale_dist'] = args.is_rescale_dist
            if args.alpha < 2:
                if 'v2_' in args.manifold:  
                    config['d_intrinsic'] = attn_setup['d_intrinsic']
                else:
                    if args.n_attn_heads == 1:
                        config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden//args.n_attn_heads - 1
                        #config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic'])-1)/np.pi)   
                        #config.sphere_radius = 1   
                    else:
                        config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden//args.n_attn_heads
                        #config['sphere_radius'] = ((np.pi**(1/config['d_intrinsic']))/np.pi)                                   
            #elif args.alpha >= 2:
            config['sphere_radius'] = attn_setup['sphere_radius'] = 1                 
        
            # mask for distance
            config['mask_val'] = attn_setup['mask_val'] = config['sphere_radius'] * np.pi
            attn_setup['sphere_radius'] = config['sphere_radius']   

        elif 'rd' in args.manifold:                            

            model_name = args.manifold + model_name
            config['is_rescale_dist'] = args.is_rescale_dist
            if args.alpha < 2:
                if 'v2_' in args.manifold:
                    config['d_intrinsic'] = attn_setup['d_intrinsic'] = 1
                else:
                    # head_dim
                    config['d_intrinsic'] = attn_setup['d_intrinsic'] = args.hidden//args.n_attn_heads  

            # if config['is_rescale_dist'] and config['alpha'] < 2:
            #     from mutils import fdm_kernel
            #     from scipy.optimize import brentq
            #     def kernel_difference(x):
            #         return fdm_kernel(x, config['alpha'], config['d_intrinsic'], 
            #         bandwidth=config['bandwidth'], is_rescaled_dist=config['is_rescale_dist']) -\
            #              fdm_kernel(x, 2, config['d_intrinsic'], bandwidth=config['bandwidth'], is_rescaled_dist=config['is_rescale_dist'])                
            #     config['intersection'] = intersection = brentq(kernel_difference, 1, 1e9)

            #     print(f'Intersection = {intersection} \n')

        # degree index
        config['a'] = attn_setup['a'] = args.a      

    elif 'sink' in model_name:
        config['n_it'] = attn_setup['n_it'] = args.n_it
        #config['bandwidth'] = attn_setup['bandwidth'] = args.bandwidth
    
    if model_name in ['rdfnsformer', 'v2_rdfnsformer']:
        from models.rdfnsformer import RDFNSformer
        model = RDFNSformer(config)               
    elif model_name == 'spfnsformer':
        from models.spfnsformer import SPFNSformer
        model = SPFNSformer(config)                                      
    elif model_name == 'sinkformer':
        from models.sinkformer import SINKformer
        model = SINKformer(config)
    elif model_name == 'dpformer':
        from models.dpformer import DPformer
        model = DPformer(config)   

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Number of parameters of the model is %d' % count_parameters(model))

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    #model = torch.nn.DataParallel(model)
    model.to(device)
    if args.fix_embed:
        if config['pretrained_model_name'] == 'glove':
            if not torch.is_tensor(pretrained_word_embeddings):
                pretrained_word_embeddings = torch.tensor(pretrained_word_embeddings)
            model.embedding = nn.Embedding.from_pretrained(nn.Parameter(pretrained_word_embeddings).to(device), freeze=True)      
        elif config['pretrained_model_name'] in ['distilbert-base-uncased', 'albert-base-v2', 'gpt2']:
            if not torch.is_tensor(pretrained_word_embeddings):
                pretrained_word_embeddings = torch.tensor(pretrained_word_embeddings)
            model.embedding =\
                 nn.Embedding.from_pretrained(nn.Parameter(pretrained_word_embeddings).to(device), freeze=True)      
            if not torch.is_tensor(pretrained_position_embeddings):
                pretrained_position_embeddings = torch.tensor(pretrained_position_embeddings)                 
            model.pos_embedding =\
                nn.Embedding.from_pretrained(nn.Parameter(pretrained_position_embeddings).to(device), freeze=True)

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
    if config['train_mask_type'] is not None:
        del config['train_mask']
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
    #optimizer = Adam(model.parameters(), lr=learning_rate)  # sinkformer
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(beta1,beta2), weight_decay=args.weight_decay)
    #optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1,beta2), weight_decay=args.weight_decay)    

    # helps estimate an arbitrarily accurate loss over either split using many batches
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
    #X, Y = get_batch('train') # fetch the very first batch
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
    #for epoch in range(epochs):    
    for epoch in tqdm(range(epochs)):
        #print('epoch: ', epoch)
        epoch_loss = 0
        epoch_accuracy = 0

        #for batch in tqdm(train_loader):
        for batch in train_loader:

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
                #if epoch + 1 < 15:
                if epoch + 1 < epochs * 3/4:
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
            #best_val_loss = epoch_val_loss
            #if iter_num > 0:
            if epoch == epochs - 1:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    #'model_args': model_args,
                    'iter_num': iter_num,
                    #'best_val_loss': best_val_loss,
                    'epoch_val_loss': epoch_val_loss,
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
    # delete
    if args.fix_embed:
        if args.pretrained_model_name in ['distilbert-base-uncased', 'albert-base-v2', 'gpt2']:
            is_match_word = torch.equal(model.embedding.weight, pretrained_word_embeddings.to(device))
            is_match_position = torch.equal(model.pos_embedding.weight, pretrained_position_embeddings.to(device))
            message = 'Embeddings params DO NOT match!' if not (is_match_word and is_match_position) else 'Embeddings params match!'
            print(message)    
        elif args.pretrained_model_name == 'glove':
            is_match = torch.equal(model.embedding.weight, pretrained_word_embeddings.to(device))
            message = 'Embeddings params DO NOT match!' if not is_match else 'Embeddings params match!'
            print(message)            