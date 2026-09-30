import argparse
import os
from datetime import datetime, timedelta
from itertools import product
from os.path import isfile, isdir
from time import sleep
from constants import MODEL_SUFFIX, DROOT, CLUSTER, PHYSICS_CONDA, RESOURCE_CONFIGS, GADI_SOURCE
from utils.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs, str_to_time, time_to_str
    
if __name__ == '__main__':
      
    parser = argparse.ArgumentParser(description='batch_submit_main.py args')   
    parser.add_argument('--is_qsub', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--nstack', type=int, default=1) 
    args = parser.parse_args()

    batch_script_name = "batch_main.py"
    script_name = "main.py"    

    is_train_others = True
    model_names = ['sinkformer', 'dpformer']  # 'dpformer'

    seeds = list(range(5))
    # seeds = list(range(2))
    # is_ops = [False,True]
    is_ops = [True]

    # MODEL HYPERPARAMETERS
    d_model = 64
    num_layers = 4
    num_heads = 8
    # num_epochs = 20
    # num_epochs = 25
    # num_epochs = 30
    # num_epochs = 35
    num_epochs = 40

    # FNS settings
    is_rescale_dist = True
    manifolds = ['rd']
    # alphas = [1.2]
    alphas = [1.2, 2]
    # alphas = [1.0, 1.4, 1.6, 1.8]
    # alphas = [1, 1.2, 1.4, 1.6, 1.8, 2]
    bandwidths = [1]
    # traning setting
    #lr = 2e-4  # v3
    #lr, lr_reduction_factor = 1.5e-4, 0.3  # v5
    #lr, lr_reduction_factor = 1.5e-4, 0.75  # v6
    #lr, lr_reduction_factor = 1.5e-4, 0.85  # v7
    #lr, lr_reduction_factor, min_lr = 2e-4, 0.75, 1.6e-4  # gscale
    #lr, lr_reduction_factor, min_lr = 2e-4, 0.7, 1.8e-4  # gscale2    

    # Resources
    nstack = args.nstack
    mem = '6GB'      
    is_use_gpu = True

    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]         
    select = 1
   
    scheduler = 'warmup_cosine'

    # scheduler = 'noam'; 
    # noam_factor = 0.35  # m3

    # scheduler = 'reduce_on_plateau'
    # lr_reduction_factor = 0.75
    # patience = 5  # r3

    for is_op in is_ops:
        ##### Original settings for reduce_on_plateau #####
        # if not is_op:
        #     lr, lr_reduction_factor, min_lr = 2e-4, 0.75, 0  # gscale3
        # else:
        #     lr, lr_reduction_factor, min_lr = 2.2e-4, 0.75, 0  # gscale3
        ##### New setting for warmup_cosine #####
        if not is_op:
            # lr, min_lr = 1e-3, 1e-5  # lr1
            # lr, min_lr = 2e-3, 2e-5  # lr2
            # lr, min_lr = 3e-3, 3e-5  # lr3
            # lr, min_lr = 4e-3, 4e-5  # lr4
            # lr, min_lr = 6e-3, 6e-5  # lr5
            # lr, min_lr = 9e-3, 4e-5  # lr6
            # lr, min_lr = 4.5e-3, 4e-5  # lr7
            # lr, min_lr = 5e-3, 5e-5  # lr8
            lr, min_lr = 4e-3, 3.5e-5  # lr9
        else:
            # lr, min_lr = 1e-3, 1e-5  # lr1
            # lr, min_lr = 2e-3, 2e-5  # lr2
            # lr, min_lr = 3e-3, 3e-5  # lr3
            # lr, min_lr = 4e-3, 4e-5  # lr4
            # lr, min_lr = 6e-3, 6e-5  # lr5
            # lr, min_lr = 9e-3, 4e-5  # lr6
            # lr, min_lr = 4.5e-3, 4e-5  # lr7
            # lr, min_lr = 5e-3, 5e-5  # lr8
            # lr, min_lr = 4e-3, 3.5e-5  # lr9
            lr, min_lr = 6e-3, 4.5e-5  # lr10

        ##### New setting for noam #####
        # if not is_op:
        #     lr = 1e-3  # lr1
        # else:
        #     lr = 1e-3  # lr1

        ##### reduce_on_plateau #####
        # if not is_op:
        #     # lr, min_lr = 2e-4, 1e-5  # lr1
        #     lr, min_lr = 3e-4, 1e-5  # lr2
        # else:
        #     # lr, min_lr = 1e-4, 1e-5  # lr1
        #     lr, min_lr = 3e-4, 1e-5  # lr2

        # single_walltime = '00:25:59' if not is_op else '00:35:00'  # 30 epochs 
        # single_walltime = '00:50:59' if not is_op else '01:05:00'  # 35 - 40 epochs
        # single_walltime = '00:40:59' if not is_op else '00:55:00'  # 35 epochs, d = 256
        # single_walltime = '00:10:59' if not is_op else '00:15:00'  # 25 epochs
        # single_walltime = '00:11:29' if not is_op else '00:12:59'  # 30 epochs
        # single_walltime = '00:13:29' if not is_op else '00:14:29'  # 35 epochs
        single_walltime = '00:15:29' if not is_op else '00:17:29'  # 40 epochs
        walltime = time_to_str(str_to_time(single_walltime) * nstack)
        # ROOT = njoin(DROOT, 'exps_gscale3')
        ROOT = njoin(DROOT, 
                     f'full_model-{scheduler}-lr10-sharp', 
                     f'l={num_layers}-h={num_heads}-d={d_model}-ep={num_epochs}'
                     )
        job_path = njoin(ROOT, 'jobs_all')

        kwargss_all = []    
        for seed in seeds:                             
                
            common_kwargs = {'seed':               seed, 
                            'is_op':               is_op,
                            'd_model':             d_model,
                            'num_layers':          num_layers,
                            'num_heads':           num_heads, 
                            'num_epochs':          num_epochs, 
                            'lr':                  lr, 
                            'scheduler':           scheduler
                            }       

            if scheduler == 'reduce_on_plateau':
                common_kwargs['lr_reduction_factor'] = lr_reduction_factor
                common_kwargs['patience'] = patience
                common_kwargs['min_lr'] = min_lr
            elif scheduler == 'warmup_cosine':
                common_kwargs['min_lr'] = min_lr
            elif scheduler == 'noam':
                common_kwargs['noam_factor'] = noam_factor

            model_root = ROOT
            
            kwargss = []            
            # FNS
            for alpha, bandwidth, manifold in product(alphas, bandwidths, manifolds):
                model_name = manifold + 'fns' +  MODEL_SUFFIX
                # model_name = 'op' + model_name if is_op else model_name
                model_dir = njoin(model_root,
                f'{model_name}-alpha={float(alpha)}-eps={float(bandwidth)}',
                f'model={seed}')
                kwargss.append({'model_name':'fns' + MODEL_SUFFIX,'alpha':alpha,'a': 0,
                                'bandwidth':bandwidth,'manifold':manifold,
                                'is_rescale_dist': is_rescale_dist})
            
            # Other models
            if is_train_others:
                for model_name in model_names:
                    # model_name = 'op' + model_name if is_op else model_name
                    if model_name == 'dp' + MODEL_SUFFIX:
                        model_name = 'dp' + MODEL_SUFFIX
                        model_dir = njoin(model_root,f'{model_name}',f'model={seed}')                
                        #if not isfile(njoin(model_dir, 'run_performance.csv')) or is_force_train:
                        kwargss.append({'model_name':'dp' + MODEL_SUFFIX})
                    # Only schedule the paper-standard Sinkformer in a non-OP sweep.
                    # if not is_op:
                    elif model_name == 'sink' + MODEL_SUFFIX:
                        for n_it in [3]:
                            kwargss.append({'model_name':'sinkformer', 'n_it':n_it, 'bandwidth':1})


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
                kwargs_qsubs[i]['source'] = GADI_SOURCE
                qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])
