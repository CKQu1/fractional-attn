import os
from datetime import datetime
from os.path import isfile, isdir
from time import sleep
from constants import DROOT, CLUSTER, PHYSICS_CONDA
from mutils import njoin, get_instance, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs
"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
      
    script_name = "batch_main.py"
    nstack = 4

    # add or change datasets here
    patch_size = 1
    #patch_size = 4
    #DATASET_NAMES = ['cifar10']  #  'pathfinder-classification'            
    DATASET_NAMES = ['mnist']
    
    n_layers = [1]
    #seeds = list(range(5))        
    seeds = [0,1]

    for n_layer in n_layers:
        ROOT = njoin(DROOT, f'{n_layer}L-ps={patch_size}-v3')
        job_path = njoin(ROOT, 'jobs_all')

        kwargss_all = []    
        for seed in seeds:
            for didx, dataset_name in enumerate(DATASET_NAMES):            
                select = 1; 
                ngpus, ncpus = 1, 1  # GPU
                #ngpus, ncpus = 0, 1  # CPU                            
                walltime = '23:59:59'
                mem = '12GB' if n_layer >= 4 else '8GB'                                              
                num_proc = ngpus if ngpus > 1 else ncpus

                for qk_share in [True, False]:
                #for qk_share in [True]:
                    #for is_op in [True, False]:
                    for is_op in [True]:

                        kwargss = []

                        if n_layer <= 4:                                                
                            for alpha in [1.2, 2]:
                            #for alpha in [1, 1.2, 1.4, 1.6, 1.8, 2]:
                                for bandwidth in [1]:                              
                                    #for manifold in ['sphere']:                                
                                    for manifold in ['rd']:
                                            kwargss.append({'model_name':'fnsvit','manifold':manifold,
                                            'alpha': alpha,'a': 0,'bandwidth':bandwidth, 'is_op':is_op}
                                            )     
                        else:                      
                            for alpha in [1.2, 2]:                            
                                for bandwidth in [1]:
                                #for bandwidth in [0.01, 1]:
                                    #for manifold in ['sphere']:                                
                                    for manifold in ['rd']:
                                        kwargss.append({'model_name':'fnsvit','manifold':manifold,
                                        'alpha': alpha,'a': 0,'bandwidth':bandwidth,'is_op':is_op}
                                        )    

                        # ----- dpvit -----
                        #kwargss.append({'model_name':'dpvit','is_op':is_op})
                        # ----- sinkvit -----
                        # for n_it in [3]:
                        #     kwargss.append({'model_name':'opsinkvit','n_it':n_it,'is_op':is_op})                            
                            
                        common_kwargs = {'seed':              seed,
                                        'qk_share':          qk_share, 
                                        'n_layers':          n_layer,
                                        'hidden_size':       48,
                                        'patch_size':        patch_size,                                                                                                                                 
                                        'weight_decay':      0
                                        }  
                        if n_layer == 1:
                            common_kwargs['lr_scheduler_type'] = 'binary'
                            #common_kwargs['max_lr'] = 1e-4
                            common_kwargs['max_lr'] = 1e-3
                            common_kwargs['min_lr'] = 1e-4

                            common_kwargs['epochs'] = 45
                            common_kwargs['n_layers'] = 1
                            common_kwargs['n_attn_heads'] = 1   
                            common_kwargs['train_bs'] = 32                                                                

                            common_kwargs['is_rescale_dist'] = True
                        elif 1 < n_layer <= 4:
                            common_kwargs['lr_scheduler_type'] = 'constant'
                            #common_kwargs['max_lr'] = 1e-4
                            common_kwargs['max_lr'] = 1e-3

                            common_kwargs['epochs'] = 50
                            common_kwargs['n_attn_heads'] = 1   
                            common_kwargs['train_bs'] = 32           
                             
                            common_kwargs['is_rescale_dist'] = True      
                        else:
                            common_kwargs['lr_scheduler_type'] = 'binary'
                            common_kwargs['max_lr'] = 1e-4
                            common_kwargs['min_lr'] = 1e-5

                            common_kwargs['epochs'] = 300
                            #common_kwargs['n_layers'] = 6                            
                            common_kwargs['n_attn_heads'] = 6
                            #common_kwargs['n_attn_heads'] = 1 if is_op else 6
                            common_kwargs['train_bs'] = 32                         
                            
                            common_kwargs['is_rescale_dist'] = True
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
            qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], path=job_path,**kwargs_qsubs[i])         