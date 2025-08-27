import os
from datetime import datetime
from itertools import product
from os.path import isfile, isdir
from time import sleep
from constants import DROOT, CLUSTER, MODEL_SUFFIX
from UTILS.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs

"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
          
    date_str = datetime.today().strftime('%Y-%m-%d')    
    batch_script_name = "batch_main.py"
    script_name = 'main.py'
    nstack = 60
    MEM_DICT = {1: '4GB', 2: '8GB', 3: '8GB', 4: '10GB', 5: '12GB', 6: '14GB'}
    #CLUSTER = 'PHYSICS'  # can manually enter here too
    q = 'l40s'  # 'l40s', 'taiji'        

    # ensembles
    seeds = list(range(5))
    # dataset
    DATASET_NAMES = ['imdb']  # 'imdb', 'glue-sst2', 'ag_news', 'emotion', 'yelp_polarity'
    # embeddings
    fix_embed = False
    if fix_embed:
        pretrained_model_name = 'glove'  # glove, distilbert-base-uncased, albert-base-v2
    # special masks
    train_mask_type = None  # 'longformer', None
    # architecture
    is_ops = [True]
    qk_shares = [True, False]
    is_rescale_dist = True
    is_resnet_scale = False
    max_len = 512
    n_layers = 3
    if n_layers < 4:
        hiddens = [8, 16 ,32 ,64]
        #hiddens = [64]
        n_attn_heads = 1
    else:                
        hiddens = [256] 
        n_attn_heads = 8    
    # training
    lr_scheduler_type = 'binary'  # 'constant'
    if lr_scheduler_type == 'binary':
        binary_ratio = 3/4  # for L-d-grid study

    # resources
    is_use_gpu = True
    select = 1
    (ngpus,ncpus) = (1,1) if is_use_gpu else (0,1)                               
    walltime = '23:59:59'
    mem = MEM_DICT[n_layers]   

    # models
    is_force_train = False
    # FNS
    #alphas = [1, 1.2, 1.4, 1.6, 1.8, 2] 
    alphas = [1.2, 2]
    #alphas = [1]
    bandwidths = [1]  # 'median'      
    manifolds = ['rd']  # 'rd', 'v2_rd', 'sphere'       
    # other types
    is_train_others = True          

    kwargss_all = []    
    for seed, dataset_name in product(seeds, DATASET_NAMES):                                                                                                         
        for hidden, qk_share, is_op in product(hiddens, qk_shares, is_ops):
            # setting up directories
            f_prefix = f'{n_layers}L'
            if is_resnet_scale:
                f_prefix += '-RR'            
            ROOT = njoin(DROOT, 'L-d-grid')
            job_path = njoin(ROOT, 'jobs_all')
            ROOT = njoin(ROOT, f'{f_prefix}-hidden={hidden}-max_len={max_len}')
            if fix_embed:
                ROOT += f'-{pretrained_model_name}'
            if train_mask_type is not None:
                ROOT += f'-mask={train_mask_type}'        
            if is_rescale_dist:
                ROOT += '-rescaled'                    

            common_kwargs = {                                  
                "seed":              seed,
                "qk_share":          qk_share,
                'is_op':             is_op,
                "is_rescale_dist":   is_rescale_dist,   
                "is_resnet_scale":   is_resnet_scale,    
                'lr_scheduler_type': lr_scheduler_type, 
                'binary_ratio':      binary_ratio if lr_scheduler_type=='binary' else None,                                                                                                                        
                "weight_decay":      0,
                'max_lr':            1e-4,
                'min_lr':            1e-5,
                'n_layers':          n_layers,
                'hidden':            hidden                            
            }  
            #common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        
            common_kwargs['max_len'] = max_len
            common_kwargs['fix_embed'] = fix_embed
            if fix_embed:
                common_kwargs["pretrained_model_name"] = pretrained_model_name
            if train_mask_type is not None:
                common_kwargs["train_mask_type"] = train_mask_type                

            if n_layers > 3:
                common_kwargs['n_attn_heads'] = n_attn_heads
                common_kwargs["epochs"] = 20
                common_kwargs["max_lr"] = 1e-4                            
                common_kwargs["min_lr"] = 1e-5
                common_kwargs["train_bs"] = 32
            else:
                common_kwargs['n_attn_heads'] = n_attn_heads
                if dataset_name.lower() in ['imdb', 'rotten_tomatoes', 'ag_news']:
                    common_kwargs["epochs"] = 25
                else:
                    common_kwargs["epochs"] = 35                                                            
                # common_kwargs["max_lr"] = 1e-3
                # common_kwargs["min_lr"] = 1e-4
                common_kwargs["max_lr"] = 1e-3
                common_kwargs["min_lr"] = 2e-4
                common_kwargs["train_bs"] = 16

            # ----- add more settings here -----                                           
            model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                                                  
            model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)

            kwargss = []
            qkv = 'qqv' if qk_share else 'qkv'
            # FNS
            for alpha, bandwidth, manifold in product(alphas, bandwidths, manifolds):
                model_name = manifold + 'fns' +  MODEL_SUFFIX
                model_name = 'op' + model_name if is_op else model_name
                model_dir = njoin(model_root,
                f'{model_name}-{dataset_name}-{qkv}-alpha={float(alpha)}-eps={float(bandwidth)}',
                f'model={seed}')
                if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                    kwargss.append({'model_name':'fnsformer','alpha':alpha,'a': 0,
                                    'bandwidth':bandwidth,'manifold':manifold})
            
            # Other models
            if is_train_others:
                model_name = 'dp' + MODEL_SUFFIX
                model_name = 'op' + model_name if is_op else model_name
                model_dir = njoin(model_root,f'{model_name}-{dataset_name}-{qkv}',f'model={seed}')                
                if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                    kwargss.append({'model_name':'dpformer'})
                # for n_it in [3]:
                #     kwargss.append({'model_name':'sinkformer','n_it':n_it,'is_op': is_op})              

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
        batch_kwargss_all.append({'arg_strss': arg_strss[:-1], 'script': script_name})

    print(f'Batched Total jobs: {len(batch_kwargss_all)} \n')

    commands, batch_script_names, pbs_array_trues, kwargs_qsubs =\
            job_setup(batch_script_name, batch_kwargss_all,
                    q=q,
                    ncpus=ncpus,
                    ngpus=ngpus,
                    select=select, 
                    walltime=walltime,
                    mem=mem,                    
                    job_path=job_path,
                    nstack=nstack,
                    cluster=CLUSTER)
    
    # for i in range(len(commands)):
    #     qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])         