import argparse
import os
from datetime import datetime, timedelta
from itertools import product
from os.path import isfile, isdir
from time import sleep
from constants import MODEL_SUFFIX, DROOT, CLUSTER, PHYSICS_CONDA, RESOURCE_CONFIGS
from UTILS.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs, str_to_time, time_to_str
"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""   
    
if __name__ == '__main__':
      
    parser = argparse.ArgumentParser(description='batch_submit_main.py args')   
    parser.add_argument('--is_qsub', type=str2bool, nargs='?', const=True, default=False) 
    args = parser.parse_args()

    batch_script_name = "batch_main.py"
    script_name = "main.py"    

    # add or change datasets here
    DATASET_NAMES = ['cifar10']  #  'cifar10', 'mnist'   

    # general settings
    #n_layers = [4]
    n_layer = 4
    seeds = list(range(5))   
    is_force_train = False   
    is_train_others = True  

    patch_size = 4
    is_preln = True  # default is True            
    qk_shares = [False]
    is_ops = [False, True]

    # FNS settings
    is_rescale_dist = True
    manifolds = ['rd']
    alphas = [1.2, 2]
    bandwidths = [1]

    # Resources
    nstack = 4
    mem = '8GB'      
    is_use_gpu = True

    cfg = RESOURCE_CONFIGS[CLUSTER][is_use_gpu]
    q, ngpus, ncpus = cfg["q"], cfg["ngpus"], cfg["ncpus"]         
    select = 1
   
    for is_op in is_ops:
        single_walltime = '01:09:59' if not is_op else '01:45:59'    
        walltime = time_to_str(str_to_time(single_walltime) * nstack)
        dirname = f'{n_layer}L-ps={patch_size}' 
        dirname = dirname + '-preln' if is_preln else dirname + '-postln'
        ROOT = njoin(DROOT, 'full_models-v6', dirname)
        job_path = njoin(ROOT, 'jobs_all')

        kwargss_all = []    
        for seed, dataset_name, qk_share in product(seeds, DATASET_NAMES, qk_shares):                              
                
            common_kwargs = {'seed':             seed,
                            'is_preln':          is_preln,  
                            'qk_share':          qk_share, 
                            'n_layers':          n_layer,
                            'hidden_size':       48,
                            'patch_size':        patch_size,                                                                                                                                 
                            'weight_decay':      0
                            }  
            if n_layer == 1:
                common_kwargs['lr_scheduler_type'] = 'binary'
                #common_kwargs['max_lr'] = 1e-4
                common_kwargs['max_lr'] = 4e-3
                common_kwargs['min_lr'] = 4e-4

                common_kwargs['epochs'] = 45
                common_kwargs['n_layers'] = 1
                common_kwargs['n_attn_heads'] = 1   
                common_kwargs['train_bs'] = 32                                                                

                common_kwargs['is_rescale_dist'] = is_rescale_dist
            else:
                common_kwargs['lr_scheduler_type'] = 'binary'
                if common_kwargs['lr_scheduler_type'] == 'binary':
                    common_kwargs['binary_ratio'] = 4/5
                # common_kwargs['max_lr'] = 5e-4  # v2
                if is_op:
                    common_kwargs['max_lr'] = 6e-4  # v3 (best for is_op True)             
                else:
                    #common_kwargs['max_lr'] = 7e-4  # v4
                    #common_kwargs['max_lr'] = 1e-3  # v5                    
                    common_kwargs['max_lr'] = 2e-3  # v6
                common_kwargs['min_lr'] = common_kwargs['max_lr'] / 10         

                common_kwargs['epochs'] = 125                                                        
                common_kwargs['n_attn_heads'] = 6
                #common_kwargs['n_attn_heads'] = 1 if is_op else 6
                common_kwargs['train_bs'] = 64                         
                
                common_kwargs['is_rescale_dist'] = is_rescale_dist

            use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                 

            model_root_dirname = structural_model_root(qk_share=qk_share, n_layers=common_kwargs['n_layers'],
                                                    n_attn_heads=common_kwargs['n_attn_heads'], hidden_size=common_kwargs['hidden_size']
                                                    )       
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
                qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])                