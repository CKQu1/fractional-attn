import os
from datetime import datetime
from itertools import product
from os.path import isfile, isdir
from time import sleep
from constants import DROOT, CLUSTER, MODEL_SUFFIX
from UTILS.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs

import re
from pathlib import Path

if __name__ == '__main__':

    date_str = datetime.today().strftime('%Y-%m-%d')    
    batch_script_name = "batch_main.py"
    script_name = 'dynamic_inference.py'
    nstack = 16

    #CLUSTER = 'PHYSICS'  # can manually enter here too
    q = 'taiji'  # 'l40s', 'taiji'   

    # resources
    is_use_gpu = False
    select = 1
    (ngpus,ncpus) = (1,1) if is_use_gpu else (0,1)                               
    walltime = '23:59:59'
    mem = '4GB'      

    # extract model_dir
    models_root = Path(njoin(DROOT, 'L-d-grid')) 
    job_path = njoin(models_root, 'inference_jobs_all')
    pattern = re.compile(r"model=\d+$")
    all_model_dirs = [str(p) for p in models_root.rglob("*") if p.is_dir() and pattern.search(str(p))]

    # isolate layers
    n_layers = 1
    all_model_dirs = [model_dir for model_dir in all_model_dirs if f'{n_layers}L-hidden' in model_dir]

    # general setting
    is_dist_based = False
    batch_size = 64
    fname = 'dist' if is_dist_based else 'prob'
    fname += f'-bs={batch_size}-inference.csv'
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
            for alpha in SELECTED_ALPHAS:
                if f'alpha={float(alpha)}' in model_dir:
                    break

        if model_dir is not None and isfile(njoin(model_dir, 'ckpt.pt'))\
             and not isfile(njoin(model_dir, fname)):
            kwargss_all.append({'model_dir': model_dir, 'batch_size': batch_size})

    # ----- submit jobs -----
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
    
    for i in range(len(commands)):
        qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])     