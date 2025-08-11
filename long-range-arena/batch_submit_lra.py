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
    nstack = 3
    is_use_gpu = True

    select = 1 
    if is_use_gpu:
        ngpus, ncpus = 1, 1  # GPU
        #mem = '12GB' if n_layer >= 4 else '8GB'
        mem = '12GB'
    else:
        ngpus, ncpus = 0, 1  # CPU                            
        #mem = '48GB' if n_layer >= 4 else '24GB'
        mem = '48GB'
    walltime = '23:59:59'                                                              
    num_proc = ngpus if ngpus > 1 else ncpus

    # add or change datasets here        
    DATASET_NAMES = ['pathfinder-classification']
    
    #n_layers = [1]
    #seeds = list(range(5))        
    seeds = [0]

    qk_shares = [True,False]
    is_ops = [True]
    manifolds = ['rd']
    alphas = [1.2, 2]
    
    # for n_layer in n_layers:
    ROOT = njoin(DROOT, f'exps')
    job_path = njoin(ROOT, 'jobs_all')

    kwargss_all = []    
    for seed in seeds:
        for didx, dataset_name in enumerate(DATASET_NAMES):            
            for qk_share in qk_shares:
                for is_op in is_ops:
                    kwargss = []

                    #if n_layer <= 4:                                                
                    for alpha in alphas:
                        for bandwidth in [1]:                                                         
                            for manifold in manifolds:
                                    kwargss.append({'model_name':'fnsformer','manifold':manifold,
                                    'alpha': alpha,'a': 0,'bandwidth':bandwidth, 'is_op':is_op}
                                    )     

                    # ----- dpformer -----
                    kwargss.append({'model_name':'dpformer','is_op':is_op})
                    # ----- sinkformer -----
                    # for n_it in [3]:
                    #     kwargss.append({'model_name':'opsinkformer','n_it':n_it,'is_op':is_op})                            
                        
                    common_kwargs = {'seed':              seed,
                                    'qk_share':          qk_share
                                    }  
                    # if n_layer == 1:
                    common_kwargs['lr_scheduler_type'] = 'constant'
                    #common_kwargs['max_lr'] = 1e-4
                    common_kwargs['max_lr'] = 1e-2
                    #common_kwargs['min_lr'] = 4e-4
                    common_kwargs['min_lr'] = None

                    common_kwargs['epochs'] = 200
                    common_kwargs['num_encoder_layers'] = None
                    common_kwargs['num_attention_heads'] = None
                    common_kwargs['train_bs'] = 32                                                                

                    common_kwargs['is_rescale_dist'] = True
                    # if num_proc > 1:
                    #     common_kwargs['grad_accum_step'] = num_proc * 2

                    use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                 

                    model_root_dirname = structural_model_root(qk_share=qk_share, num_encoder_layers=common_kwargs['num_encoder_layers'],
                                                            n_attn_heads=common_kwargs['num_attention_heads']
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