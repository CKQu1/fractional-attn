import os
import re
from itertools import product
from os.path import isfile, isdir
from pathlib import Path

from constants import DROOT, MODEL_SUFFIX, CLUSTER
from UTILS.mutils import njoin, collect_model_dirs, load_model_files, structural_model_root
from qsub_parser import add_common_kwargs

"""
The following are all the experiments to be run from `batch_submit_main.py`
"""

# ----- model training experiments (full) -----
def train_exps_full():

    script_name = 'main.py'    
    MEM_DICT = {1: '4GB', 2: '8GB', 3: '8GB', 4: '8GB', 5: '10GB', 6: '12GB'}
    #CLUSTER = 'PHYSICS'  # can manually enter here too
    #q = 'l40s'  # 'l40s', 'taiji', 'h100'        

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
    is_rescale_dist = True
    is_resnet_scale = False
    max_len = 512

    # -------------------- CHANGE HERE --------------------
    n_layers = 6 
    # -----------------------------------------------------

    if n_layers < 4:
        hiddens = [8, 16 ,32 ,64]
        #hiddens = [64]
        n_attn_heads = 1

        is_ops = [True]
        qk_shares = [True, False]

        # model root dir
        ROOT = njoin(DROOT, 'L-d-grid-v2')
        job_path = njoin(ROOT, 'jobs_all')        
    else:                
        hiddens = [256] 
        n_attn_heads = 8    

        is_ops = [False, True]
        qk_shares = [False]

        # model root dir
        #ROOT = njoin(DROOT, 'full_models')
        ROOT = njoin(DROOT, 'full_models-v3')
        job_path = njoin(ROOT, 'jobs_all')          
    # training
    lr_scheduler_type = 'binary'  # 'constant'
    if lr_scheduler_type == 'binary':
        binary_ratio = 3/4  # for L-d-grid study and 6 layered models

    # resources
    is_use_gpu = True
    if is_use_gpu:
        if CLUSTER == 'GADI':
            q = 'gpuvolta'
            ngpus, ncpus = 1, 12  # GPU       
        elif CLUSTER == 'PHYSICS':
            q = 'l40s'
            ngpus, ncpus = 1, 1  # GPU
    else:
        if CLUSTER == 'GADI':
            q = 'normal'
        elif CLUSTER == 'PHYSICS':
            q = 'taiji'         
        ngpus, ncpus = 0, 1  # CPU             

    nstack = 1                               
    walltime = '01:29:59'  # for nstack = 1
    mem = MEM_DICT[n_layers]   
    select = 1

    # models
    is_force_train = False
    # FNS
    if n_layers == 1:
        alphas = [1, 1.2, 1.4, 1.6, 1.8, 2] 
    else:
        alphas = [1.2, 2]
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
                'n_attn_heads':      n_attn_heads                             
            }  
            #common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        
            common_kwargs['max_len'] = max_len
            common_kwargs['fix_embed'] = fix_embed
            if fix_embed:
                common_kwargs["pretrained_model_name"] = pretrained_model_name
            if train_mask_type is not None:
                common_kwargs["train_mask_type"] = train_mask_type                

            if n_layers > 3:
                common_kwargs["epochs"] = 20
                common_kwargs["max_lr"] = 1e-4                            
                common_kwargs["min_lr"] = 1e-5
                common_kwargs["train_bs"] = 32
            else:
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

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack

# ----- model training experiments (layers 1, 2, 3) -----
def train_exps_hyperparam(): 

    script_name = 'main.py'
    nstack = 1
    MEM_DICT = {1: '4GB', 2: '8GB', 3: '8GB', 4: '8GB', 5: '10GB', 6: '12GB'}
    #CLUSTER = 'PHYSICS'  # can manually enter here too
    #q = 'l40s'  # 'l40s', 'taiji', 'h100'        

    # ensembles
    seeds = list(range(1))
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

    # -------------------- CHANGE HERE --------------------
    n_layers = 1  # 1, 2, 3
    # -----------------------------------------------------

    if n_layers < 4:
        hiddens = [8, 16 ,32 ,64]
        n_attn_heads = 1

        is_ops = [True]
        qk_shares = [True, False]

        # model root dir
        ROOT = njoin(DROOT, 'L-d-grid-v2')
        job_path = njoin(ROOT, 'jobs_all')        
    else:                
        hiddens = [256] 
        n_attn_heads = 8    

        is_ops = [False, True]
        qk_shares = [False]

        # model root dir
        #ROOT = njoin(DROOT, 'full_models')
        ROOT = njoin(DROOT, 'full_models-v2')
        job_path = njoin(ROOT, 'jobs_all')          
    # training
    lr_scheduler_type = 'binary'  # 'constant'
    if lr_scheduler_type == 'binary':
        binary_ratio = 3/4  # for L-d-grid study and 6 layered models

    # resources
    is_use_gpu = True
    if is_use_gpu:
        if CLUSTER == 'GADI':
            q = 'gpuvolta'
            ngpus, ncpus = 1, 12  # GPU       
        elif CLUSTER == 'PHYSICS':
            q = 'l40s'
            ngpus, ncpus = 1, 1  # GPU
    else:
        if CLUSTER == 'GADI':
            q = 'normal'
        elif CLUSTER == 'PHYSICS':
            q = 'taiji'         
        ngpus, ncpus = 0, 1  # CPU             

    select = 1                           
    #walltime = '01:29:59'  # for nstack = 1
    walltime = '00:11:59'
    mem = MEM_DICT[n_layers]   

    # models
    is_force_train = False
    # FNS
    if n_layers == 1:
        alphas = [1, 1.2, 1.4, 1.6, 1.8, 2] 
    else:
        alphas = [1.2, 2]
    bandwidths = [1]  # 'median'      
    manifolds = ['rd']  # 'rd', 'v2_rd', 'sphere'       
    # other types
    is_train_others = False    

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
                'n_attn_heads':      n_attn_heads                             
            }  
            #common_kwargs['max_len'] = MAX_LENS_DICT[dataset_name] if max_len is None else max_len                        
            common_kwargs['max_len'] = max_len
            common_kwargs['fix_embed'] = fix_embed
            if fix_embed:
                common_kwargs["pretrained_model_name"] = pretrained_model_name
            if train_mask_type is not None:
                common_kwargs["train_mask_type"] = train_mask_type                

            if n_layers > 3:
                common_kwargs["epochs"] = 20
                common_kwargs["max_lr"] = 1e-4                            
                common_kwargs["min_lr"] = 1e-5
                common_kwargs["train_bs"] = 32
            else:
                if dataset_name.lower() in ['imdb', 'rotten_tomatoes', 'ag_news']:
                    common_kwargs["epochs"] = 25
                else:
                    common_kwargs["epochs"] = 35                                                            
                # common_kwargs["max_lr"] = 1e-3
                # common_kwargs["min_lr"] = 1e-4
                if n_layers == 1:
                    common_kwargs["max_lr"] = 1e-3
                    common_kwargs["min_lr"] = 2e-4
                elif n_layers == 2:
                    common_kwargs["max_lr"] = 5e-4
                    common_kwargs["min_lr"] = 1e-4   
                elif n_layers == 3:
                    common_kwargs["max_lr"] = 2e-4
                    common_kwargs["min_lr"] = 4e-5                                        
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

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack

# ----- dynamic inference for pretrained models -----
def dynamic_inference_exps():
    script_name = 'dynamic_inference.py'
    nstack = 4

    #CLUSTER = 'PHYSICS'  # can manually enter here too  

    # ---------- CHANGE HERE ----------
    # isolate layers
    n_layers = 1    

    # general setting
    is_normal_mode = True
    if not is_normal_mode: 
        is_dist_based = False
        batch_size = 64

    #models_root = Path(njoin(DROOT, 'L-d-grid'))
    models_root = Path(njoin(DROOT, 'L-d-grid-v2')) 
    job_path = njoin(models_root, 'inference_jobs_all')        
    # ---------------------------------

    # resources
    is_use_gpu = True
    if is_use_gpu:
        if CLUSTER == 'GADI':
            q = 'gpuvolta'
            ngpus, ncpus = 1, 12  # GPU       
        elif CLUSTER == 'PHYSICS':
            q = 'l40s'
            ngpus, ncpus = 1, 1  # GPU
    else:
        if CLUSTER == 'GADI':
            q = 'normal'
        elif CLUSTER == 'PHYSICS':
            q = 'taiji'         
        ngpus, ncpus = 0, 1  # CPU  

    select = 1                             
    walltime = '00:08:59'
    mem = '4GB'      

    # extract model_dir
    pattern = re.compile(r"model=\d+$")
    all_model_dirs = [str(p) for p in models_root.rglob("*") if p.is_dir() and pattern.search(str(p))]
    all_model_dirs = [model_dir for model_dir in all_model_dirs if f'{n_layers}L-hidden' in model_dir]

    fnames =[]  # if these files do not exist, then run
    if not is_normal_mode:
        fname = 'dist' if is_dist_based else 'prob'
        fname += f'-bs={batch_size}-inference.csv'
        fnames.append(fname)
    else:
        fnames += ['bs=1-train_inference.csv', 'bs=1-test_inference.csv']
    fnames.append('ckpt.pt')

    # FNS setting
    is_op = True
    manifold = 'rd'

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
        for fname in fnames:
            is_run_exp = is_run_exp and not isfile(fname)
        if is_run_exp:
            # non-normal mode (random masking exp)
            if not is_normal_mode:
                kwargss_all.append({'model_dir': model_dir, 'batch_size': batch_size,
                                    'is_normal_mode': is_normal_mode, 'is_dist_based': is_dist_based})  
            # normal mode (batch-size 1)
            else:
                kwargss_all.append({'model_dir': model_dir, 
                                    'is_normal_mode': is_normal_mode})

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack


# ----- attention graphs for pretrained models -----
def attn_graph_exps():

    script_name = 'attn_graph_v2.py'
    nstack = 1

    #CLUSTER = 'PHYSICS'  # can manually enter here too
    q = 'l40s'  # 'l40s', 'taiji'   

    # resources
    is_use_gpu = False
    select = 1
    (ngpus,ncpus) = (1,1) if is_use_gpu else (0,1)                               
    walltime = '23:59:59'
    mem = '4GB'      

    # extract model_dir
    models_root = njoin(DROOT, 'L-d-grid', '1L-hidden=8-max_len=512-rescaled', 
                        'config_qqv', 'imdb', 'layers=1-heads=1-qqv')
    job_path = njoin(DROOT, 'L-d-grid', 'graph_jobs_all')

    # -------------------------------------

    # Get model setting from dir
    models_root = models_root.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(models_root) if 'former' in dirname])  

    # select based on bandwidth    
    DCT_ALL = collect_model_dirs(models_root, suffix=MODEL_SUFFIX)
    model_types = list(DCT_ALL.keys())
    dp_types = [model_type for model_type in model_types if 'dp' in model_type]

    fns_type = 'rdfnsformer'
    SELECTED_ALPHAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # None
    SELECTED_EPSS = [1]  # None
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
    SELECTED_SEEDS = [0]

    # ----- general setting -----
    use_same_token = False

    kwargss_all = []
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
                    pretrained_model_name = config['pretrained_model_name'] if fix_embed else False                    
                    spectrum_file = f'attn_graph-{model_name}-seed={seed}-alpha={alpha}\
                        -pretrained_embd={pretrained_model_name}-same_token={use_same_token}.npz'
                    is_add = isfile(njoin(model_dir, 'ckpt.pt')) and not isfile(njoin(model_dir, spectrum_file)) and seed in seeds_all
                    if is_add:
                        kwargss_all.append({'model_dir': model_dir, 'use_same_token': use_same_token})
        
        ##### other model types #####
        if DCT_ALL['opdpformer'].shape[0] > 0:
            seeds_all = DCT_ALL['opdpformer'].loc[0,'seeds']
        else:
            continue
        for seed in SELECTED_SEEDS:
            model_dir = njoin(DCT_ALL['opdpformer'].loc[0,'model_dir'], f'model={seed}')
            attn_setup, config, _, _ = load_model_files(model_dir)
            spectrum_file = f'attn_graph-dpformer-seed={seed}.npz'
            is_add = isfile(njoin(model_dir, 'ckpt.pt')) and not isfile(njoin(model_dir, spectrum_file)) and seed in seeds_all  
            if is_add:   
                kwargss_all.append({'model_dir': model_dir, 'use_same_token': use_same_token})

    return kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack