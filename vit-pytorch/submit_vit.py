import os
from datetime import datetime
from os.path import isfile, isdir
from time import sleep
from constants import *
from mutils import njoin, get_instance, structural_model_root
from qsub_parser import *
"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
    
    # ----- System -----
    system = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS'    
    date_str = datetime.today().strftime('%Y-%m-%d')    
    #script_name = "ddp_main.py"  
    script_name = "main.py"
    nstack = 1  

    # add or change datasets here
    DATASET_NAMES = ['pathfinder-classification']  #  'pathfinder-classification'            
    
    ROOT = njoin(DROOT, 'full-models-v3')    
    job_path = njoin(ROOT, 'jobs_all', date_str)

    #instances = list(range(5))    
    instances = [0]
    #model_sizes = ['small', 'large']
    model_sizes = ['large']

    for model_size in model_sizes:
        kwargss_all = []    
        for instance in instances:
            for didx, dataset_name in enumerate(DATASET_NAMES):            
                select = 1; ngpus, ncpus = 1, 0                            
                walltime = '23:59:59'
                mem = '32GB'                                              
                num_proc = ngpus if ngpus > 1 else ncpus

                #for qk_share in [True, False]:
                for qk_share in [False]:
                    kwargss = []

                    if model_size == 'small':
                        for model_name in ['opfnsvit']:                        
                            for alpha in [1.2, 1.6, 2]:                            
                                for bandwidth in [0.01, 0.1, 0.5, 1]:
                                    for manifold in ['rd', 'sphere']:                                
                                        kwargss.append({'model_name':model_name,'manifold':manifold,'alpha': alpha,'a': 0,'bandwidth':bandwidth})       
                    else:
                        for model_name in ['opfnsvit']:                        
                            for alpha in [1.2, 2]:                            
                                for bandwidth in [1]:
                                    for manifold in ['sphere']:                                
                                        kwargss.append({'model_name':model_name,'manifold':manifold,'alpha': alpha,'a': 0,'bandwidth':bandwidth})                          

                    #kwargss.append({'model_name':'opfnsvit','manifold':'sphere','alpha': 1.2,'a': 0,'bandwidth':1})

                    # kwargss.append({'model_name':'dpvit'})
                    # for n_it in [3]:
                    #     kwargss.append({'model_name':'sinkvit','n_it':n_it})
                        
                    common_kwargs = {'instance':          instance,
                                    'qk_share':          qk_share, 
                                    'hidden_size':       48,                                                                                                                                 
                                    'weight_decay':      0,
                                    #'lr_scheduler_type': 'constant',
                                    #'lr_scheduler_type': 'cosine',
                                    'lr_scheduler_type': 'binary',
                                    'max_lr':            1e-4,
                                    'min_lr':            1e-5
                                    }  

                    if model_size == 'small':
                        common_kwargs['epochs'] = 200
                        common_kwargs['n_layers'] = 1
                        common_kwargs['n_attn_heads'] = 1   
                        common_kwargs['train_bs'] = 32                         
                    else:
                        common_kwargs['epochs'] = 300 if dataset_name == 'cifar10' else 100
                        common_kwargs['n_layers'] = 6
                        common_kwargs['n_attn_heads'] = 8
                        common_kwargs['train_bs'] = 32                         
                        common_kwargs['patch_size'] = 16
                        
                    # if num_proc > 1:
                    #     common_kwargs['grad_accum_step'] = num_proc * 2

                    use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                 

                    model_root_dirname = structural_model_root(qk_share=qk_share, n_layers=common_kwargs['n_layers'],
                                                            n_attn_heads=common_kwargs['n_attn_heads'], hidden_size=common_kwargs['hidden_size']
                                                            )       
                    model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)
                
            
                    for idx in range(len(kwargss)):
                        # function automatically creates dir
                        kwargss[idx]["dataset"] = dataset_name    
                        kwargss[idx]['model_root'] = model_root
                    
                    kwargss = add_common_kwargs(kwargss, common_kwargs)
                    kwargss_all += kwargss

        print(f'Total jobs: {len(kwargss_all)} \n')              

        commands, script_names, pbs_array_trues, kwargs_qsubs =\
                job_setup(script_name, kwargss_all,
                        ncpus=ncpus,
                        ngpus=ngpus,
                        select=select, 
                        walltime=walltime,
                        mem=mem,
                        job_path=job_path,
                        nstack=nstack,
                        system=system)
        
        for i in range(len(commands)):
            qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])         