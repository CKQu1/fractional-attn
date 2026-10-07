import math
import os
import re
from itertools import product
from os.path import isfile, isdir
from pathlib import Path

from constants import DROOT, MODEL_SUFFIX, CLUSTER, RESOURCE_CONFIGS
from UTILS.mutils import njoin, collect_model_dirs, load_model_files, structural_model_root, create_model_dir
from qsub_parser import add_common_kwargs, str_to_time, time_to_str

"""
The following are all the experiments to be run from `batch_submit_main.py`
"""

# ----- model training experiments (full) -----
def train_exps_full(nstack, manifold):

    print(f'manifold = {manifold}')
    print('hello')

    script_name = 'main.py'    
    #CLUSTER = 'PHYSICS'  # can manually enter here too
    #q = 'l40s'  # 'l40s', 'taiji', 'h100'        

    # -------------------- CHANGE HERE --------------------
    # ensembles
    seeds = list(range(5))
    # seeds = [0]
    # dataset
    DATASET_NAMES = ['imdb']  # 'imdb', 'ag_news', 'emotion'
    # embeddings
    fix_embed = False
    if fix_embed:
        pretrained_model_name = 'glove'  # glove, distilbert-base-uncased, albert-base-v2
    # special masks
    train_mask_type = None  # 'longformer', None
    # architecture
    is_rescale_dist = True
    is_resnet_scale = False
    # max_len = 2048
    max_len = 1024
    # max_len = 512
    # max_len = 256
    # max_len = 128
    
    n_layers, n_attn_heads = 6, 8
    qkv_bias = False
    hiddens = [256]     
    d_ffn_ratio = 2  # ratio of ffn width to d_k
    weight_decay = 0
    # weight_decay = 1e-4
    max_lr = 1e-4 if manifold == 'rd' else 1.2e-4

    is_ops = [False,True]
    qk_shares = [False]

    # FNA setting
    a = 0

    # model root dir
    # ROOT = njoin(DROOT, 'trial-full-sphere-v6')
    #ROOT = njoin(DROOT, f'full_models-qkv_bias={qkv_bias}-gscalev5-3')
    # ROOT = njoin(DROOT, f'full_models-new_mask-a={a}')
    # ROOT = njoin(DROOT, f'data_split-full_models-new_mask-a={a}')
    # ROOT = njoin(DROOT, 
    #              f'relu-nsplit-v1mask-v6scale-qkv_bias={qkv_bias}-'
    #              f'a={a}-ratio={d_ffn_ratio}-wd={weight_decay}-lr={max_lr}',
    #              )

    ROOT = njoin(DROOT, f'full_models-v1mask-v7scale')  # same as above setting

    job_path = njoin(ROOT, 'jobs_all')    

    # models
    is_force_train = False
    is_train_others = False  # other types
    is_train_fns = True
    other_model_prefixes = ['dp', 'sink']  # 'dp', 'sink'

    # FNS
    # alphas = [1.2, 2]
    alphas = [1, 1.4, 1.6, 1.8]
    bandwidths = [1]      
    # manifolds = ['rd']  # 'rd', 'v2_rd', 'sphere'       

    kwargss_all = []
    walltimes_all, mems_all = [], []
    for seed, dataset_name in product(seeds, DATASET_NAMES):      

        # training
        epochs = 20
        lr_scheduler_type = 'binary'  # 'constant'
        if lr_scheduler_type == 'binary':
            binary_ratio = 3/4  # for L-d-grid study and 6 layered models

        # resources
        is_use_gpu = True
        cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
        q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]              
                
        if manifold == 'rd':
            if dataset_name == 'imdb':      
                # single_walltime, mem = '01:20:59', '12GB'  # for n = 512
                default_single_walltime, default_mem = '01:40:59', '12GB'  # for n = 512, ffn_hidden = 1024
                # single_walltime, mem = '02:10:59', '12GB'  # for n = 512, ffn_hidden = 1024, sinkformer
                # single_walltime, mem = '02:30:59', '36GB'  # for n = 1024, ffn_hidden = 512
                # single_walltime, mem = '01:15:59', '8GB'  # for n = 256, ffn_hidden = 512
#                 fns_default_single_walltime = dp_default_single_walltime =\
# sink_default_single_walltime = opfns_default_single_walltime =\
# opdp_default_single_walltime = opsink_default_single_walltime =\
# default_single_walltime
                if max_len == 512:
                    dp_default_single_walltime = '00:41:00'; opdp_default_single_walltime = '00:47:00'
                    fns_default_single_walltime = '01:25:00'; opfns_default_single_walltime = '01:30:00'
                    sink_default_single_walltime = '02:00:00'; opsink_default_single_walltime = '02:07:00' 
                elif max_len == 1024:
                    dp_default_single_walltime = '02:30:00'; opdp_default_single_walltime = '02:30:00'
                    fns_default_single_walltime = '05:30:00'; opfns_default_single_walltime = '05:30:00'
                    sink_default_single_walltime = '07:15:00'; opsink_default_single_walltime = '07:30:00' 
                elif max_len == 256:
                    dp_default_single_walltime = '00:21:00'; opdp_default_single_walltime = '00:27:00'
                    fns_default_single_walltime = '01:05:00'; opfns_default_single_walltime = '01:10:00'
                    sink_default_single_walltime = '01:40:00'; opsink_default_single_walltime = '01:47:00' 
                elif max_len == 128:
                    dp_default_single_walltime = '00:21:00'; opdp_default_single_walltime = '00:27:00'
                    fns_default_single_walltime = '01:05:00'; opfns_default_single_walltime = '01:10:00'
                    sink_default_single_walltime = '01:40:00'; opsink_default_single_walltime = '01:47:00' 
                elif max_len == 2048:
                    dp_default_single_walltime = '07:20:00'; opdp_default_single_walltime = '08:20:00'
                    fns_default_single_walltime = '18:30:00'; opfns_default_single_walltime = '19:10:00'
                    sink_default_single_walltime = '27:00:00'; opsink_default_single_walltime = '28:45:00' 
            elif dataset_name == 'ag_news':
                # default_single_walltime, default_mem = '03:50:59', '12GB'  # max_len = 512
                if max_len == 512:
                    default_mem = '12GB'
                    dp_default_single_walltime = '02:50:59'; opdp_default_single_walltime = '02:50:59'
                    fns_default_single_walltime = '03:45:59'; opfns_default_single_walltime = '03:55:59'
                    sink_default_single_walltime = '05:00:59'; opsink_default_single_walltime = '05:20:59'
                elif max_len == 256:
                    default_mem = '8GB'
                    dp_default_single_walltime = '01:00:59'; opdp_default_single_walltime = '01:25:59'
                    fns_default_single_walltime = '01:35:59'; opfns_default_single_walltime = '02:00:59'
                    sink_default_single_walltime = '02:00:59'; opsink_default_single_walltime = '02:30:59'
            else:
                raise ValueError(f'No train_exps_full resources configured for {manifold=}, {dataset_name=}')
        elif manifold == 'sphere':
            default_single_walltime, default_mem = '01:55:59', '12GB'  # for n = 512
        else:
            raise ValueError(f'No train_exps_full resources configured for {manifold=}')

        # Edit this table to request different resources for each model/op pair.
        # The True block is for the op* variants submitted with is_op=True.
        model_resources = {
            False: {
                'fnsformer':  {'single_walltime': fns_default_single_walltime, 'mem': default_mem},
                'dpformer':   {'single_walltime': dp_default_single_walltime, 'mem': default_mem},
                'sinkformer': {'single_walltime': sink_default_single_walltime, 'mem': default_mem},
            },
            True: {
                'fnsformer':  {'single_walltime': opfns_default_single_walltime, 'mem': default_mem},
                'dpformer':   {'single_walltime': opdp_default_single_walltime, 'mem': default_mem},
                'sinkformer': {'single_walltime': opsink_default_single_walltime, 'mem': default_mem},
            },
        }

        select = 1

        for hidden, qk_share, is_op in product(hiddens, qk_shares, is_ops):
            # setting up directories
            f_prefix = f'{n_layers}L'
            if is_resnet_scale:
                f_prefix += '-RR'            

            MODELS_ROOT = njoin(ROOT, f'{f_prefix}-hidden={hidden}-max_len={max_len}')
            if fix_embed:
                MODELS_ROOT += f'-{pretrained_model_name}'
            if train_mask_type is not None:
                MODELS_ROOT += f'-mask={train_mask_type}'        
            if is_rescale_dist:
                MODELS_ROOT += '-rescaled'                    

            common_kwargs = {                                  
                "seed":              seed,
                "qk_share":          qk_share,
                "qkv_bias":          qkv_bias,
                'is_op':             is_op,
                "is_rescale_dist":   is_rescale_dist,   
                "is_resnet_scale":   is_resnet_scale,    
                'lr_scheduler_type': lr_scheduler_type, 
                'binary_ratio':      binary_ratio if lr_scheduler_type=='binary' else None,                                                                                                                        
                "weight_decay":      weight_decay,
                'n_layers':          n_layers,
                'hidden':            hidden,
                # 'ffn_hidden':        hidden,
                'ffn_hidden':        hidden*d_ffn_ratio,
                'n_attn_heads':      n_attn_heads                             
            }  
            #common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        
            common_kwargs['max_len'] = max_len
            common_kwargs['fix_embed'] = fix_embed
            if fix_embed:
                common_kwargs["pretrained_model_name"] = pretrained_model_name
            if train_mask_type is not None:
                common_kwargs["train_mask_type"] = train_mask_type                

            # if n_layers > 3:
            common_kwargs["epochs"] = epochs
            if max_len <= 512:
                common_kwargs["train_bs"] = 32
            elif max_len <= 1024:
                common_kwargs["train_bs"] = 8
            elif max_len <= 2048:
                common_kwargs["train_bs"] = 4
            if manifold == 'rd':
                common_kwargs["max_lr"] = max_lr  # full_model-v2, full_model-v3, gscale, gscalev2, gscalev5
                #common_kwargs["max_lr"] = 2e-4  # gscalev5-2                               
                common_kwargs["min_lr"] = common_kwargs["max_lr"] / 10
            elif manifold == 'sphere':
                common_kwargs["max_lr"] = 1.2e-4                             
                common_kwargs["min_lr"] = common_kwargs["max_lr"] / 10

            # ----- add more settings here -----                                           
            model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                                                  
            model_root = njoin(MODELS_ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)

            kwargss = []
            kwargss_walltimes, kwargss_mems = [], []
            qkv = 'qqv' if qk_share else 'qkv'

            def append_model_kwargs(model_kwargs, model_type):
                resource = model_resources[is_op][model_type]
                kwargss.append(model_kwargs)
                kwargss_walltimes.append(
                    time_to_str(str_to_time(resource['single_walltime']) * nstack)
                )
                kwargss_mems.append(resource['mem'])

            # FNS
            # for alpha, bandwidth, manifold in product(alphas, bandwidths, manifolds):
            if is_train_fns:
                for alpha, bandwidth in product(alphas, bandwidths):
                    manifold_prefix = 'sp' if manifold == 'sphere' else manifold
                    model_name = manifold_prefix + 'fns' +  MODEL_SUFFIX
                    model_name = 'op' + model_name if is_op else model_name
                    model_dir = njoin(model_root,
                    f'{model_name}-{dataset_name}-{qkv}-alpha={float(alpha)}-eps={float(bandwidth)}',
                    f'model={seed}')
                    if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                        append_model_kwargs({'model_name':'fnsformer','alpha':alpha,'a': a,
                                             'bandwidth':bandwidth,'manifold':manifold},
                                            'fnsformer')
            
            # Other models
            if is_train_others:
                for model_prefix in other_model_prefixes:
                    model_name = model_prefix + MODEL_SUFFIX
                    model_name = 'op' + model_name if is_op else model_name

                    if model_prefix == 'dp':
                        model_dir = njoin(model_root,f'{model_name}-{dataset_name}-{qkv}',f'model={seed}')                
                        if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                            append_model_kwargs({'model_name': 'dpformer'}, 'dpformer')
                    elif model_prefix == 'sink':
                        for n_it in [3]:
                            attn_setup = {'model_name':model_name,'n_it':n_it,'dataset_name':dataset_name,
                                          'qk_share':qk_share}
                            model_dir = njoin(create_model_dir(model_root, **attn_setup)[0], f'model={seed}')                
                            if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                                append_model_kwargs({'model_name':'sinkformer','n_it':n_it}, 'sinkformer')

            for idx in range(len(kwargss)):
                # function automatically creates dir
                kwargss[idx]["dataset_name"] = dataset_name    
                kwargss[idx]['model_root'] = model_root
            
                kwargss = add_common_kwargs(kwargss, common_kwargs)
                
            kwargss_all += kwargss
            walltimes_all += kwargss_walltimes
            mems_all += kwargss_mems

    assert len(kwargss_all) == len(walltimes_all) == len(mems_all), 'resources must match jobs'
    return kwargss_all, script_name, q, ncpus, ngpus, select, walltimes_all, mems_all, job_path, nstack


# ----- model training experiments (layers 1, 2, 3) -----
def train_exps_hyperparam(nstack): 

    script_name = 'main.py'    
    MEM_DICT = {1: '4GB', 2: '8GB', 3: '8GB', 4: '8GB', 5: '10GB', 6: '12GB'}
    #CLUSTER = 'PHYSICS'  # can manually enter here too
    #q = 'l40s'  # 'l40s', 'taiji', 'h100'        

    # ensembles
    # seeds = list(range(1))
    seeds = list(range(5))
    # seeds = list(range(1,5))
    # dataset
    DATASET_NAMES = ['imdb']  # 'imdb', 'glue-sst2', 'ag_news', 'emotion', 'yelp_polarity'
    # embeddings
    fix_embed = False
    if fix_embed:
        pretrained_model_name = 'glove'  # glove, distilbert-base-uncased, albert-base-v2
    # special masks
    train_mask_type = None  # 'longformer', None
    # architecture
    is_rescale_dist = True
    is_resnet_scale = False
    max_len = 512
    # max_len = 256

    # -------------------- CHANGE HERE --------------------
    n_layers = 1  # 1, 2
    # -----------------------------------------------------

    # if n_layers < 4:
    # hiddens = [8, 16 ,32 ,64]
    hiddens = [128, 256]
    n_attn_heads = 1
    ffn_hidden = 256  # default

    is_ops = [True]
    # qk_shares = [False, True]
    qk_shares = [True]

    # model root dir
    #ROOT = njoin(DROOT, 'L-d-grid-v2')
    # ROOT = njoin(DROOT, 'L-d-grid-v4')
    # ROOT = njoin(DROOT, 'L-d-grid-v1mask-v7scaling')
    ROOT = njoin(DROOT, 'L-d-grid-v1mask-v7scaling-v2')
    # ROOT = njoin(DROOT, 'L-d-grid-v1mask-v7scaling-v3')
    job_path = njoin(ROOT, 'jobs_all')        
        
    # training
    lr_scheduler_type = 'binary'  # 'constant'
    if lr_scheduler_type == 'binary':
        # binary_ratio = 3/4  # for L-d-grid study, 25 epochs
        # binary_ratio = 2/3  # for L-d-grid study, 15 epochs
        binary_ratio = 3/4  # for L-d-grid study, 20 epochs

    # resources
    is_use_gpu = True
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]            
                        
    if n_layers == 1:    
        # single_walltime = '00:35:59'  # imdb
        # single_walltime = '00:12:59'  # imdb  is_op = True
        single_walltime = '00:15:59'
    elif n_layers == 2:    
        single_walltime = '00:45:59'        

    walltime = time_to_str(str_to_time(single_walltime) * nstack)    
    mem = MEM_DICT[n_layers]  
    select = 1 

    # models
    is_force_train = False
    # FNS
    # alphas = [1, 1.2, 1.4, 1.6, 1.8, 2] 
    alphas = [1.2, 2]
    # alphas = [1, 1.4, 1.6, 1.8]
    bandwidths = [1]  # 'median'      
    manifolds = ['rd']  # 'rd', 'v2_rd', 'sphere'       
    # other types
    is_train_others = False    
    other_model_prefixes = ['dp', 'sink']  # 'dp', 'sink'

    kwargss_all = []    
    for seed, dataset_name in product(seeds, DATASET_NAMES):                                                                                                         
        for hidden, qk_share, is_op in product(hiddens, qk_shares, is_ops):
            # setting up directories
            f_prefix = f'{n_layers}L'
            if is_resnet_scale:
                f_prefix += '-RR'            

            MODELS_ROOT = njoin(ROOT, f'{f_prefix}-hidden={hidden}-max_len={max_len}')
            if fix_embed:
                MODELS_ROOT += f'-{pretrained_model_name}'
            if train_mask_type is not None:
                MODELS_ROOT += f'-mask={train_mask_type}'        
            if is_rescale_dist:
                MODELS_ROOT += '-rescaled'                    

            common_kwargs = {                                  
                "seed":              seed,
                "qk_share":          qk_share,
                'is_op':             is_op,
                "is_rescale_dist":   is_rescale_dist,   
                "is_resnet_scale":   is_resnet_scale,    
                'lr_scheduler_type': lr_scheduler_type, 
                'binary_ratio':      binary_ratio if lr_scheduler_type=='binary' else None,                                                                                                                        
                "weight_decay":      0,
                'n_layers':          n_layers,
                'hidden':            hidden,
                'ffn_hidden':        ffn_hidden,
                'n_attn_heads':      n_attn_heads                             
            }  
            #common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        
            common_kwargs['max_len'] = max_len
            common_kwargs['fix_embed'] = fix_embed
            if fix_embed:
                common_kwargs["pretrained_model_name"] = pretrained_model_name
            if train_mask_type is not None:
                common_kwargs["train_mask_type"] = train_mask_type                

            # smaller model settings
            if dataset_name.lower() in ['imdb', 'rotten_tomatoes', 'ag_news']:
                # common_kwargs["epochs"] = 25  # binary_ratio = 3/4
                # common_kwargs["epochs"] = 15  # binary_ratio = 2/3
                common_kwargs["epochs"] = 20  # binary_ratio = 3/4
            else:
                common_kwargs["epochs"] = 35                                                            
            if n_layers == 1:
                # for 20, 25 epochs
                if hidden <= 64:
                    common_kwargs["max_lr"] = 1e-3
                    common_kwargs["min_lr"] = 2e-4    
                elif hidden <= 256:
                    common_kwargs["max_lr"] = 5e-4
                    common_kwargs["min_lr"] = 1e-4  
                elif hidden <= 512:
                    common_kwargs["max_lr"] = 2.5e-4
                    common_kwargs["min_lr"] = 5e-5  
                # for 15 epochs
                # common_kwargs["max_lr"] = 2e-3
                # common_kwargs["min_lr"] = 4e-4     
            elif n_layers == 2:
                # common_kwargs["max_lr"] = 5e-4
                # common_kwargs["min_lr"] = 1e-4  
                common_kwargs["max_lr"] = 1e-4
                common_kwargs["min_lr"] = 2e-5  
            common_kwargs["train_bs"] = 16

            # ----- add more settings here -----                                           
            model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                                                  
            model_root = njoin(MODELS_ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)

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
                for model_prefix in other_model_prefixes:
                    if model_prefix == 'dp':
                        model_name = 'dp' + MODEL_SUFFIX
                        model_name = 'op' + model_name if is_op else model_name
                        model_dir = njoin(model_root,f'{model_name}-{dataset_name}-{qkv}',f'model={seed}')                
                        if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                            kwargss.append({'model_name':'dpformer'})
                    elif model_prefix == 'sink':
                        model_name = 'sink' + MODEL_SUFFIX
                        model_name = 'op' + model_name if is_op else model_name
                        model_dir = njoin(model_root,f'{model_name}-{dataset_name}-{qkv}',f'model={seed}')                
                        if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                            for n_it in [3]:
                                kwargss.append({'model_name':'sinkformer','n_it':n_it})              

            for idx in range(len(kwargss)):
                # function automatically creates dir
                kwargss[idx]["dataset_name"] = dataset_name    
                kwargss[idx]['model_root'] = model_root
            
            kwargss = add_common_kwargs(kwargss, common_kwargs)
            kwargss_all += kwargss

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack

# ----- dynamic inference for small pretrained models -----
def dynamic_inference_small(nstack, is_dist_based=False):
    script_name = 'dynamic_inference.py' 

    # ---------- CHANGE HERE ----------   
    n_layers = 1  # isolate layers
    is_force_run = True

    # is_normal_mode = True  # for gettings accuracies corresponding to sequence lengths
    is_normal_mode = False  # for deconstructing attention graph
    if is_normal_mode:
        max_len_adj = 512
    else:
        # is_dist_based = kwargs.get('is_dist_based')
        batch_size = 64
        max_len = 512

    # models_root = Path(njoin(DROOT, 'L-d-grid'))
    # models_root = Path(njoin(DROOT, 'L-d-grid-v2'))
    # models_root = Path(njoin(DROOT, 'L-d-grid-v1mask-v7scaling'))
    models_root = Path(njoin(DROOT, 'L-d-grid-v1mask-v7scaling-v2'))
    job_path = njoin(models_root, 'inference_jobs_all')        
    # ---------------------------------

    # resources
    #CLUSTER = 'PHYSICS'  # can manually enter here too  
    is_use_gpu = True
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]  

    select = 1                             
    # single_walltime = '00:03:05'
    if is_dist_based:
        single_walltime = '00:09:29'
    else:
        single_walltime = '00:04:05'
    walltime = time_to_str(str_to_time(single_walltime) * nstack)    
    mem = '4GB'      

    # extract model_dir
    pattern = re.compile(r"model=\d+$")
    all_model_dirs = [str(p) for p in models_root.rglob("*") if p.is_dir() and pattern.search(str(p))]
    
    ##### Include all ds ######
    # all_model_dirs = [model_dir for model_dir in all_model_dirs\
    #                    if f'{n_layers}L-hidden' in model_dir and\
    #                       f'max_len={max_len}' in model_dir]

    ##### Select ds #####
    selected_ds = [8, 16, 32, 64]
    # selected_ds = [8]
    all_model_dirs = [
        model_dir
        for model_dir in all_model_dirs
        if any(f'{n_layers}L-hidden={d}-' in model_dir for d in selected_ds)
        and f'max_len={max_len}' in model_dir
    ]

    fnames = []  # if these files do not exist, then run
    if not is_normal_mode:
        fname = 'dist' if is_dist_based else 'prob'
        fname += f'-bs={batch_size}-inference.csv'
        fnames.append(fname)
    else:
        fnames += [f'train_inference-bs=1-len={max_len_adj}.csv', f'test_inference-bs=1-len={max_len_adj}.csv']
    fnames.append('ckpt.pt')

    # FNS setting
    is_op = True
    manifold = 'rd'

    SELECTED_ALPHAS = [1.2, 2.0]    
    # SELECTED_ALPHAS = []

    kwargss_all = []
    fns_type = manifold + 'fns' + MODEL_SUFFIX
    if is_op:
        fns_type = 'op' + fns_type
    for model_dir in all_model_dirs:
        is_fns = f'/{fns_type}' in model_dir
        if is_fns:
            # isolate alphas from SELECTED_ALPHAS
            if not any(f'alpha={float(alpha)}' in model_dir for alpha in SELECTED_ALPHAS):
                continue
        # else:
        #     continue

        is_run_exp = model_dir is not None
        for fname in fnames[:-1]:
            is_run_exp = is_run_exp and not isfile(njoin(model_dir, fname))  # data files
        is_run_exp = is_run_exp and isfile(njoin(model_dir, fnames[-1]))     # ckpt.pt
        if is_run_exp or is_force_run:
            # non-normal mode (random masking exp)
            if not is_normal_mode:
                kwargss_all.append({'model_dir': model_dir, 'batch_size': batch_size,
                                    'is_normal_mode': is_normal_mode, 'is_dist_based': is_dist_based})  
            # normal mode (batch-size 1)
            else:
                kwargss_all.append({'model_dir': model_dir, 
                                    'is_normal_mode': is_normal_mode, 'max_len_adj': max_len_adj})

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack


# ----- dynamic inference for large pretrained models -----
def dynamic_inference_full(nstack):
    script_name = 'dynamic_inference.py'

    # ---------- CHANGE HERE ----------    
    n_layers = 6
    is_normal_mode = True
    #max_len_adj = 512  # default
    max_len_adj = 1024
    # FNS setting
    is_op = True
    manifold = 'rd'

    models_root = Path(njoin(DROOT, 'full_models-v2')) 
    job_path = njoin(models_root, 'inference_jobs_all')        
    # ---------------------------------

    # resources
    #CLUSTER = 'PHYSICS'  # can manually enter here too  
    is_use_gpu = True
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]  

    select = 1                             
    single_walltime = '00:15:25'
    walltime = time_to_str(str_to_time(single_walltime) * nstack)    
    mem = '4GB'      

    # extract model_dir
    pattern = re.compile(r"model=\d+$")
    all_model_dirs = [str(p) for p in models_root.rglob("*") if p.is_dir() and pattern.search(str(p))]
    all_model_dirs = [model_dir for model_dir in all_model_dirs if f'{n_layers}L-hidden' in model_dir]

    # if these files do not exist, then run
    fnames = [f'train_inference-bs=1-len={max_len_adj}.csv', f'test_inference-bs=1-len={max_len_adj}.csv']
    fnames.append('ckpt.pt')


    SELECTED_ALPHAS = [1.2, 2.0]    
    #SELECTED_EPSS = [1.0]  

    kwargss_all = []
    fns_type = manifold + 'fns' + MODEL_SUFFIX
    if is_op:
        fns_type = 'op' + fns_type
    for model_dir in all_model_dirs:
        is_fns = f'/{fns_type}' in model_dir
        if is_fns:
            # isolate alphas from SELECTED_ALPHAS
            if not any(f'alpha={float(alpha)}' in model_dir for alpha in SELECTED_ALPHAS):
                continue

        is_run_exp = model_dir is not None
        for fname in fnames[:-1]:
            is_run_exp = is_run_exp and not isfile(njoin(model_dir, fname))  # data files
        is_run_exp = is_run_exp and isfile(njoin(model_dir, fnames[-1]))     # ckpt.pt
        if is_run_exp:
            # normal mode (batch-size 1)
            kwargss_all.append({'model_dir': model_dir, 
                                'is_normal_mode': is_normal_mode, 'max_len_adj': max_len_adj})

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack



# ----- attention graphs for pretrained models -----
def attn_graph_exps(nstack, script_name):

    assert script_name in ['attn_graph_v2.py', 'attn_graph_final.py'], f'{script_name} incorrect' 

    #CLUSTER = 'PHYSICS'  # can manually enter here too
    # q = 'l40s'  # 'l40s', 'taiji'   

    # resources
    is_use_gpu = False
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]      
    select = 1        

    # ROOT_ALL = njoin(DROOT, 'L-d-grid-v2')
    # ROOT_ALL = njoin(DROOT, 'L-d-grid-v1mask-v7scaling')
    ROOT_ALL = njoin(DROOT, 'L-d-grid-v1mask-v7scaling-v2')
    if script_name == 'attn_graph_v2.py':  # need to run for all L = 1
        # single_walltime = '00:18:29'
        # single_walltime = '00:20:29' 
        single_walltime = '00:30:29'       

        # ----- general setting -----
        ds = [8, 16, 32, 64]
        # ds = [8]
        fns_type = 'rdfnsformer'
        SELECTED_ALPHAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # None
        # SELECTED_ALPHAS = [1.0]
        SELECTED_SEEDS = list(range(5))
        # SELECTED_SEEDS = list(range(1))

        use_same_token = True
        generated_files = ['spectrum_file.npz', 'andrew_results_1.npz', 'andrew_results_2.npz']
        job_path = njoin(ROOT_ALL, 'spectral_jobs_all')

        is_force_run = False

    elif script_name == 'attn_graph_final.py':  # selectively run
        single_walltime = '00:03:15'

        # ----- general setting -----
        ds = [8, 16]
        fns_type = 'rdfnsformer'
        SELECTED_ALPHAS = [1.2, 2.0]  # None
        SELECTED_SEEDS = list(range(5))
        # SELECTED_SEEDS = list(range(1))

        bdwth = 0.1
        generated_files = ['attn_graph_results.npz']
        job_path = njoin(ROOT_ALL, 'graph_jobs_all')

        is_force_run = False
    
    walltime = time_to_str(str_to_time(single_walltime) * nstack)
    mem = '4GB'      

    SELECTED_EPSS = [1]  # None

    kwargss_all = []
    for didx, d in enumerate(ds):
        models_root = njoin(ROOT_ALL, f'1L-hidden={d}-max_len=512-rescaled', 
                            'config_qqv', 'imdb', 'layers=1-heads=1-qqv')

        # -------------------------------------

        # Get model setting from dir
        models_root = models_root.replace('\\','') 

        # select based on bandwidth    
        DCT_ALL = collect_model_dirs(models_root, suffix=MODEL_SUFFIX)
        model_types = list(DCT_ALL.keys())
        dp_types = [model_type for model_type in model_types\
                     if 'dp' in model_type or 'sink' in model_type]

        for model_type in model_types:
            if fns_type in model_type:            
                df_model = DCT_ALL[model_type].dropna(subset='alpha')
                if SELECTED_ALPHAS is not None:
                    # ----- filter alphas -----
                    df_model = df_model[df_model['alpha'].isin(SELECTED_ALPHAS)]
                    # ------------------------
                    df_model.reset_index(drop=True, inplace=True)
                if SELECTED_EPSS is not None:
                    # ----- filter alphas -----
                    df_model = df_model[df_model['bandwidth'].isin(SELECTED_EPSS)]
                    # ------------------------
                    df_model.reset_index(drop=True, inplace=True)
                break

        # ----- fns setting -----
        alphas = sorted(df_model.loc[:,'alpha'].unique())  # small to large
        epss = sorted(df_model.loc[:,'bandwidth'].unique())  

        for eps_idx, eps in enumerate(epss):

            ##### FNS model type #####
            for ii in range(df_model.shape[0]):
                ensembles = df_model.loc[ii,'ensembles']            
                if ensembles > 0 and df_model.loc[ii,'bandwidth'] == eps:
                    seeds_all = df_model.loc[ii,'seeds']
                    for seed in SELECTED_SEEDS:
                        #attn_setup, config, _, _ = load_model_files(model_dir)

                        model_dir = njoin(df_model.loc[ii,'model_dir'], f'model={seed}')
                        attn_setup, config, _, _ = load_model_files(model_dir)
                        model_name, alpha, fix_embed = attn_setup['model_name'], attn_setup['alpha'], attn_setup['fix_embed']
                        # pretrained_model_name = config['pretrained_model_name'] if fix_embed else False                    
                        is_add = isfile(njoin(model_dir, 'ckpt.pt')) and seed in seeds_all
                        for generated_file in generated_files:
                            is_add = is_add and not isfile(njoin(model_dir, generated_file))
                        # print(f'is_add = {is_add}')
                        if is_add or is_force_run:
                            kwargs_all = {'model_dir': model_dir}
                            if script_name == 'attn_graph_v2.py':
                                kwargs_all['use_same_token'] = use_same_token
                            elif script_name == 'attn_graph_final.py':
                                # if attn_setup['alpha'] >= 2:
                                #     bdwth = math.sqrt(d)**2
                                # else:
                                #     bdwth = (d**0.5 / (2**(1/d) - 1))**attn_setup['alpha']
                                kwargs_all['bdwth'] = bdwth
                            kwargss_all.append(kwargs_all)
            
            ##### other model types #####
            other_model_types = ['opdpformer', 'opsinkformer']
            for other_model_type in other_model_types:
                if DCT_ALL[other_model_type].shape[0] > 0:
                    seeds_all = DCT_ALL[other_model_type].loc[0,'seeds']
                else:
                    continue
                for seed in SELECTED_SEEDS:
                    model_dir = njoin(DCT_ALL[other_model_type].loc[0,'model_dir'], f'model={seed}')
                    attn_setup, config, _, _ = load_model_files(model_dir)
                    is_add = isfile(njoin(model_dir, 'ckpt.pt')) and seed in seeds_all
                    for generated_file in generated_files:
                        is_add = is_add and not isfile(njoin(model_dir, generated_file))
                    if is_add or is_force_run:   
                        kwargs_all = {'model_dir': model_dir}
                        if script_name == 'attn_graph_v2.py':
                            kwargs_all['use_same_token'] = use_same_token
                        elif script_name == 'attn_graph_final.py':
                            kwargs_all['bdwth'] = bdwth
                        kwargss_all.append(kwargs_all)

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack

# ----- hyperparam analysis for datasets -----
def train_exps_hyperdataset(nstack): 

    script_name = 'main.py'    
    MEM_DICT = {1: '4GB', 2: '8GB', 3: '8GB', 4: '8GB', 5: '10GB', 6: '12GB'}
    #CLUSTER = 'PHYSICS'  # can manually enter here too
    #q = 'l40s'  # 'l40s', 'taiji', 'h100'        

    # ensembles
    seeds = list(range(5))
    # dataset
    DATASET_NAMES = ['imdb']  # 'imdb', 'glue-sst2', 'ag_news', 'emotion', 'yelp_polarity'
    trainset_pcts = [0.1, 0.2, 0.3, 0.4]
    # trainset_pcts = [0.3, 0.4]
    
    # embeddings
    fix_embed = False
    if fix_embed:
        pretrained_model_name = 'glove'  # glove, distilbert-base-uncased, albert-base-v2
    # special masks
    train_mask_type = None  # 'longformer', None
    # architecture
    is_rescale_dist = True
    is_resnet_scale = False
    max_len = 512
    # max_len = 256

    # -------------------- CHANGE HERE --------------------
    n_layers = 1  # 1, 2
    # -----------------------------------------------------

    # if n_layers < 4:
    hiddens = [8, 16 ,32 ,64]
    n_attn_heads = 1
    ffn_hidden = 256  # default

    is_ops = [True]
    # qk_shares = [False, True]
    qk_shares = [True]     
        
    # training
    lr_scheduler_type = 'binary'  # 'constant'
    if lr_scheduler_type == 'binary':
        # binary_ratio = 3/4  # for L-d-grid study, 25 epochs
        # binary_ratio = 2/3  # for L-d-grid study, 15 epochs
        binary_ratio = 3/4  # for L-d-grid study, 20 epochs

    # resources
    is_use_gpu = True
    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]            
                        
    if n_layers == 1:    
        # single_walltime = '00:35:59'  # imdb
        single_walltime = '00:12:59'  # imdb  is_op = True
    elif n_layers == 2:    
        single_walltime = '00:45:59'        

    walltime = time_to_str(str_to_time(single_walltime) * nstack)    
    mem = MEM_DICT[n_layers]  
    select = 1 

    # models
    is_force_train = False
    # FNS
    alphas = [1.2, 2]
    bandwidths = [1]  # 'median'      
    manifolds = ['rd']  # 'rd', 'v2_rd', 'sphere'       
    # other types
    is_train_others = False    
    other_model_prefixes = ['dp', 'sink']  # 'dp', 'sink'

    kwargss_all = []    
    for seed, dataset_name, trainset_pct in product(seeds, DATASET_NAMES, trainset_pcts):   

        # model root dir
        ROOT = njoin(DROOT, f'L-d-grid-trainset_pct={trainset_pct}')
        job_path = njoin(ROOT, 'jobs_all')   

        for hidden, qk_share, is_op in product(hiddens, qk_shares, is_ops):
            # setting up directories
            f_prefix = f'{n_layers}L'
            if is_resnet_scale:
                f_prefix += '-RR'            

            MODELS_ROOT = njoin(ROOT, f'{f_prefix}-hidden={hidden}-max_len={max_len}')
            if fix_embed:
                MODELS_ROOT += f'-{pretrained_model_name}'
            if train_mask_type is not None:
                MODELS_ROOT += f'-mask={train_mask_type}'        
            if is_rescale_dist:
                MODELS_ROOT += '-rescaled'                    

            common_kwargs = {                                  
                "seed":              seed,
                "trainset_pct":      trainset_pct,
                "qk_share":          qk_share,
                'is_op':             is_op,
                "is_rescale_dist":   is_rescale_dist,   
                "is_resnet_scale":   is_resnet_scale,    
                'lr_scheduler_type': lr_scheduler_type, 
                'binary_ratio':      binary_ratio if lr_scheduler_type=='binary' else None,                                                                                                                        
                "weight_decay":      0,
                'n_layers':          n_layers,
                'hidden':            hidden,
                'ffn_hidden':        ffn_hidden,
                'n_attn_heads':      n_attn_heads                             
            }  
            #common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        
            common_kwargs['max_len'] = max_len
            common_kwargs['fix_embed'] = fix_embed
            if fix_embed:
                common_kwargs["pretrained_model_name"] = pretrained_model_name
            if train_mask_type is not None:
                common_kwargs["train_mask_type"] = train_mask_type                

            # smaller model settings
            if dataset_name.lower() in ['imdb', 'rotten_tomatoes', 'ag_news']:
                # common_kwargs["epochs"] = 25  # binary_ratio = 3/4
                # common_kwargs["epochs"] = 15  # binary_ratio = 2/3
                common_kwargs["epochs"] = 20  # binary_ratio = 3/4
            else:
                common_kwargs["epochs"] = 35                                                            
            if n_layers == 1:
                # for 20, 25 epochs
                common_kwargs["max_lr"] = 1e-3
                common_kwargs["min_lr"] = 2e-4    
                # for 15 epochs
                # common_kwargs["max_lr"] = 2e-3
                # common_kwargs["min_lr"] = 4e-4     
            elif n_layers == 2:
                # common_kwargs["max_lr"] = 5e-4
                # common_kwargs["min_lr"] = 1e-4  
                common_kwargs["max_lr"] = 1e-4
                common_kwargs["min_lr"] = 2e-5  
            common_kwargs["train_bs"] = 16

            # ----- add more settings here -----                                           
            model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                                                  
            model_root = njoin(MODELS_ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)

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
                for model_prefix in other_model_prefixes:
                    if model_prefix == 'dp':
                        model_name = 'dp' + MODEL_SUFFIX
                        model_name = 'op' + model_name if is_op else model_name
                        model_dir = njoin(model_root,f'{model_name}-{dataset_name}-{qkv}',f'model={seed}')                
                        if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                            kwargss.append({'model_name':'dpformer'})
                    elif model_prefix == 'sink':
                        model_name = 'sink' + MODEL_SUFFIX
                        model_name = 'op' + model_name if is_op else model_name
                        model_dir = njoin(model_root,f'{model_name}-{dataset_name}-{qkv}',f'model={seed}')                
                        if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                            for n_it in [3]:
                                kwargss.append({'model_name':'sinkformer','n_it':n_it})              

            for idx in range(len(kwargss)):
                # function automatically creates dir
                kwargss[idx]["dataset_name"] = dataset_name    
                kwargss[idx]['model_root'] = model_root
            
            kwargss = add_common_kwargs(kwargss, common_kwargs)
            kwargss_all += kwargss

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack