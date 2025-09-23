import argparse
import os
from datetime import datetime, timedelta
from itertools import product
from os.path import isfile, isdir
from time import sleep
from constants import MODEL_SUFFIX, DROOT, CLUSTER, PHYSICS_CONDA, RESOURCE_CONFIGS, GADI_SOURCE_2
from utils.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs, str_to_time, time_to_str
    
if __name__ == '__main__':
      
    parser = argparse.ArgumentParser(description='batch_submit_main.py args')   
    parser.add_argument('--is_qsub', type=str2bool, nargs='?', const=True, default=False) 
    args = parser.parse_args()

    batch_script_name = "batch_main.py"
    script_name = "main.py"    

    is_train_others = True

    seeds = list(range(5))
    is_ops = [False,True]

    # FNS settings
    is_rescale_dist = True
    manifolds = ['rd']
    alphas = [1.2, 2]
    bandwidths = [1]
    # traning setting
    #lr = 2e-4  # v3
    #lr, lr_reduction_factor = 1.5e-4, 0.3  # v5
    #lr, lr_reduction_factor = 1.5e-4, 0.75  # v6
    #lr, lr_reduction_factor = 1.5e-4, 0.85  # v7
    #lr, lr_reduction_factor, min_lr = 2e-4, 0.75, 1.6e-4  # gscale
    #lr, lr_reduction_factor, min_lr = 2e-4, 0.7, 1.8e-4  # gscale2    

    # Resources
    nstack = 5
    mem = '9GB'      
    is_use_gpu = True

    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]         
    select = 1
   
    for is_op in is_ops:
        if not is_op:
            lr, lr_reduction_factor, min_lr = 2e-4, 0.75, 0  # gscale3
        else:
            lr, lr_reduction_factor, min_lr = 2.2e-4, 0.75, 0  # gscale3
        single_walltime = '00:25:59' if not is_op else '00:40:00'  # 30 epochs    
        walltime = time_to_str(str_to_time(single_walltime) * nstack)
        ROOT = njoin(DROOT, 'exps_gscale3')
        job_path = njoin(ROOT, 'jobs_all')

        kwargss_all = []    
        for seed in seeds:                              
                
            common_kwargs = {'seed':               seed, 
                            'is_op':               is_op,
                            'lr':                  lr,
                            'lr_reduction_factor': lr_reduction_factor}                                                          
            model_root = ROOT
            
            kwargss = []            
            # FNS
            for alpha, bandwidth, manifold in product(alphas, bandwidths, manifolds):
                model_name = manifold + 'fns' +  MODEL_SUFFIX
                model_name = 'op' + model_name if is_op else model_name
                model_dir = njoin(model_root,
                f'{model_name}-alpha={float(alpha)}-eps={float(bandwidth)}',
                f'model={seed}')
                kwargss.append({'model_name':'fns' + MODEL_SUFFIX,'alpha':alpha,'a': 0,
                                'bandwidth':bandwidth,'manifold':manifold,
                                'is_rescale_dist': is_rescale_dist})
            
            # Other models
            if is_train_others:
                model_name = 'dp' + MODEL_SUFFIX
                model_name = 'op' + model_name if is_op else model_name
                model_dir = njoin(model_root,f'{model_name}',f'model={seed}')                
                #if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                kwargss.append({'model_name':'dp' + MODEL_SUFFIX})
                # for n_it in [3]:
                #     kwargss.append({'model_name':'sinkformer','n_it':n_it,'is_op': is_op})      


            for idx in range(len(kwargss)):
                # function automatically creates dir  
                kwargss[idx]['model_root'] = model_root
            
            kwargss = add_common_kwargs(kwargss, common_kwargs)
            kwargss_all += kwargss
    
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
        
        if args.is_qsub:
            print(f'----- SUBMITTING ----- \n')
            for i in range(len(commands)):
                # use different source
                kwargs_qsubs[i]['source'] = GADI_SOURCE_2
                qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])                