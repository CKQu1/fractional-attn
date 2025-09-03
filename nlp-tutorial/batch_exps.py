from os.path import isfile, isdir

from constants import DROOT, MODEL_SUFFIX
from UTILS.mutils import njoin, collect_model_dirs, load_model_files

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
