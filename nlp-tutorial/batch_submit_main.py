import os
from datetime import datetime
from os.path import isfile, isdir
from time import sleep
from constants import DROOT, CLUSTER
from utils.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs

from constants import DATASET_NAMES, MAX_LENS, MAX_LENS_DICT

"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
      
    date_str = datetime.today().strftime('%Y-%m-%d')    
    script_name = "batch_main.py"
    #nstack = 44
    nstack = 20

    #seeds = list(range(5))
    seeds = list(range(10))
    #seeds = [0,1]
    
    n_layers = 3
    #n_layers = 6
    #n_layers = 4
    #if n_layers == 1:
    if n_layers < 4:
        model_size = 'small'  
        #alphas = []
        #alphas = [1,1.2,1.4,1.6,1.8]
        alphas = [1,1.2,1.4,1.6,1.8,2]        
        #alphas = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
        #alphas = [1.2,1.6,2]
        bandwidths = [1]
        manifolds = ['v2_rd']  # ['rd', 'sphere', 'v2_rd']

        hidden = 64

        fix_embed = False
        is_ops = [True]
        qk_shares = [True, False]
        #qk_shares = [True]
        is_rescale_dist = True
        #max_len = 2048
        #max_len = 511
        max_len = None
        #train_mask_type = 'longformer'
        train_mask_type = None

        #pretrained_model_name = 'gpt2'
        #pretrained_model_name = 'albert-base-v2'
        #pretrained_model_name = 'distilbert-base-uncased'
        pretrained_model_name = 'glove' 

        is_train_others = True
    else:
        model_size = 'large'
        #alphas = [1.2, 1.6, 2]
        #alphas = [1.2, 2]
        alphas = [1,1.2,1.4,1.6,1.8,2]
        bandwidths = [1]      
        manifolds = ['v2_rd']  # ['rd', 'v2_rd']

        fix_embed = False
        is_ops = [False, True]
        qk_shares = [True, False]
        #is_rescale_dist = False        
        is_rescale_dist = True

        hidden = 256
        max_len = None

        is_train_others = True
        #is_train_others = False

    model_sizes = [model_size]            
    if fix_embed:
        ROOT =  njoin(DROOT, f'{n_layers}L-hidden={hidden}-max_len={max_len}-{pretrained_model_name}') 
    else:
        ROOT = njoin(DROOT, f'{n_layers}L-hidden={hidden}-max_len={max_len}')
    if train_mask_type is not None:
        ROOT += f'-mask={train_mask_type}'        
    if is_rescale_dist:
        ROOT += '-rescaled'        
    job_path = njoin(ROOT, 'jobs_all')

    for model_size in model_sizes:
        kwargss_all = []    
        for seed in seeds:
            for didx in [1]:
            #for didx in [0]:

                dataset_name = DATASET_NAMES[didx]
                
                # CPUs
                #select = 1; ngpus, ncpus = 0, 8; mem = '24GB'  # imdb          
                #select = 1; ngpus, ncpus = 0, 16; mem = '32GB'  # rotten_tomatoes
                # GPUs
                if model_size == 'large':
                    select = 1; ngpus, ncpus = 1, 1; mem = '10GB'  # 6L8H imdb (512 len)                      
                else:
                    select = 1; ngpus, ncpus = 1, 1; mem = '6GB'  # 1L1H imdb (512 len)                                                                   
                walltime = '23:59:59'                    

                for qk_share in qk_shares:

                    for is_op in is_ops:

                        kwargss = []

                        for alpha in alphas:
                            for bandwidth in bandwidths:
                                for manifold in manifolds:
                                    kwargss.append({'model_name':'fnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':manifold,
                                                    'is_op': is_op})

                        if is_train_others:
                            kwargss.append({'model_name':'dpformer','is_op': is_op})

                            # for n_it in [3]:
                            #     kwargss.append({'model_name':'sinkformer','n_it':n_it,'is_op': is_op})

                        common_kwargs = {                                  
                            "seed":              seed,
                            "qk_share":          qk_share,
                            "is_rescale_dist":   is_rescale_dist,
                            #"hidden":            hidden,
                            #"hidden":           32,                          
                            #"train_bs":          32,                                          
                            #"lr_scheduler_type": "constant",                           
                            #"lr_scheduler_type": "binary",                                       
                            "weight_decay":      0,
                            'max_lr':            1e-4,
                            'min_lr':            1e-5                            
                        }  
                        #common_kwargs['max_len'] = MAX_LENS[didx] if max_len is None else max_len
                        common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        

                        #common_kwargs['n_attn_heads'] = 1 if is_op else 8
                        #common_kwargs['n_attn_heads'] = 8
     
                        if model_size == 'large':
                            common_kwargs["n_layers"] = n_layers
                            common_kwargs['n_attn_heads'] = 8
                            common_kwargs["hidden"] = hidden

                            common_kwargs["epochs"] = 20
                            #common_kwargs['fix_embed'] = False

                            common_kwargs['lr_scheduler_type'] = 'binary'
                            common_kwargs["max_lr"] = 1e-4                            
                            common_kwargs["min_lr"] = 1e-5
                            common_kwargs["train_bs"] = 32

                        elif model_size == 'small':
                            common_kwargs["pretrained_model_name"] = pretrained_model_name

                            common_kwargs["n_layers"] = n_layers
                            common_kwargs['n_attn_heads'] = 1
                            #common_kwargs["n_attn_heads"] = 1                             
                            #common_kwargs["epochs"] = 15
                            common_kwargs["epochs"] = 25
                            #common_kwargs["epochs"] = 30
                            common_kwargs['fix_embed'] = fix_embed                            

                            # common_kwargs['lr_scheduler_type'] = 'constant'
                            # #common_kwargs["max_lr"] = 3e-4
                            # common_kwargs["max_lr"] = 1.5e-4

                            common_kwargs['lr_scheduler_type'] = 'binary'
                            # common_kwargs["max_lr"] = 1e-3
                            # common_kwargs["min_lr"] = 1e-4
                            common_kwargs["max_lr"] = 1e-3
                            common_kwargs["min_lr"] = 2e-4
                            common_kwargs["train_bs"] = 16

                            common_kwargs["hidden"] = hidden
                            if train_mask_type is not None:
                                common_kwargs["train_mask_type"] = train_mask_type

                        # add more settings here                                           

                        #use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                                        
                        model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                          
                        #model_root = njoin(ROOT, config_file.split('.')[0], dataset_name, model_root_dirname)
                        model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)
                            
                        for idx in range(len(kwargss)):
                            # function automatically creates dir
                            kwargss[idx]["dataset"] = dataset_name    
                            kwargss[idx]['model_root'] = model_root
                        
                        kwargss = add_common_kwargs(kwargss, common_kwargs)
                        kwargss_all += kwargss

        print(f'Total jobs: {len(kwargss_all)} \n')      

        batch_kwargss_all = []
        kwargsss = [kwargss_all[i:i+nstack] for i in range(0, len(kwargss_all), nstack)]
        for kwargss in kwargsss:
            arg_strss = ''
            for kwargs in kwargss:
                arg_strss += ",".join("=".join((str(k),str(v))) for k,v in kwargs.items()) + ';'
            batch_kwargss_all.append({'arg_strss': arg_strss[:-1]})
  
        print(f'Batched Total jobs: {len(batch_kwargss_all)} \n')

        commands, script_names, pbs_array_trues, kwargs_qsubs =\
                job_setup(script_name, batch_kwargss_all,
                        ncpus=ncpus,
                        ngpus=ngpus,
                        select=select, 
                        walltime=walltime,
                        mem=mem,
                        job_path=job_path,
                        nstack=nstack,
                        cluster=CLUSTER)
        
        for i in range(len(commands)):
            qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])         