import argparse
import os
from datetime import datetime
from itertools import product
from os.path import isfile, isdir
from time import sleep
from constants import DROOT, CLUSTER, PHYSICS_CONDA
from mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs
"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
      
    parser = argparse.ArgumentParser(description='batch_submit_main.py args')   
    parser.add_argument('--is_qsub', type=str2bool, nargs='?', const=True, default=False) 
    args = parser.parse_args()

    batch_script_name = "batch_main.py"
    script_name = 'main.py'      
    nstack = 1
    is_use_gpu = True

    select = 1 
    if is_use_gpu:
        if CLUSTER == 'GADI':
            q = 'gpuvolta'
            ngpus, ncpus = 1, 12  # GPU            
        else:
            ngpus, ncpus = 1, 1  # GPU
        #mem = '12GB' if n_layer >= 4 else '8GB'
        mem = '10GB'
    else:
        if CLUSTER == 'GADI':
            q = 'normal'
        ngpus, ncpus = 0, 1  # CPU                            
        #mem = '48GB' if n_layer >= 4 else '24GB'
        mem = '48GB'
    walltime = '8:59:59'                                                              
    num_proc = ngpus if ngpus > 1 else ncpus

    # add or change datasets here        
    DATASET_NAMES = ['pathfinder-classification']
    apples_to_apples = True
    p_drop = 0.1  # dropout rate for embeddings, attn and hidden layers
    
    #seeds = list(range(5))        
    seeds = [0]

    # architecture settings
    qk_shares = [False]
    is_ops = [False,True]
    manifolds = ['rd']
    alphas = [1.2, 2]
    is_rescale_dist = True
    #is_preln = False
    is_preln = True
    
    # train settings
    lr_scheduler_type = 'binary'
    if lr_scheduler_type == 'binary':
        binary_ratio = 3/4

    # for n_layer in n_layers:
    ROOT = njoin(DROOT, 'exps_preLN' if is_preln else 'exps_postLN')
    job_path = njoin(ROOT, 'jobs_all')

    kwargss_all = []    
    for seed, dataset_name in product(seeds,DATASET_NAMES):
        for qk_share, is_op in product(qk_shares, is_ops):
            kwargss = []

            # ----- dpformer -----
            # kwargss.append({'model_name':'dpformer'})

            # ----- fnsformer -----                                           
            for alpha in alphas:
                for bandwidth in [1]:                                                         
                    for manifold in manifolds:
                            kwargss.append({'model_name':'fnsformer','manifold':manifold,
                            'alpha': alpha,'a': 0,'bandwidth':bandwidth, 
                            'is_rescale_dist': is_rescale_dist}
                            )     

            # ----- sinkformer -----
            # for n_it in [3]:
            #     kwargss.append({'model_name':'opsinkformer','n_it':n_it,'is_op':is_op})                            
                
            common_kwargs = {'seed':              seed,
                            'qk_share':           qk_share,
                            'is_op':              is_op,
                            'is_preln':           is_preln,
                            'apples_to_apples':   apples_to_apples,
                            'hidden_dropout_prob':p_drop,
                            'encoder_dropout_prob':p_drop,
                            'attention_probs_dropout_prob':p_drop
                            }  

            if apples_to_apples:
                # common_kwargs['lr_scheduler_type'] = 'constant'
                # #common_kwargs['max_lr'] = 2e-4
                # common_kwargs['max_lr'] = 2.5e-4

                common_kwargs['lr_scheduler_type'] = lr_scheduler_type
                if lr_scheduler_type == 'binary':
                    common_kwargs['binary_ratio'] = binary_ratio
                #common_kwargs['max_lr'] = 2e-4
                common_kwargs['max_lr'] = 2.5e-4
                common_kwargs['max_lr'] = 5e-5

                common_kwargs['train_bs'] = common_kwargs['eval_bs'] = 128                     

            # if num_proc > 1:
            #     common_kwargs['grad_accum_step'] = num_proc * 2

            use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                 

            # model_root_dirname = structural_model_root(qk_share=qk_share, num_encoder_layers=common_kwargs['num_encoder_layers'],
            #                                            n_attn_heads=common_kwargs['num_attention_heads']
            #                                            )  
            model_root_dirname = structural_model_root(qk_share=qk_share, num_encoder_layers='default',
                                                       n_attn_heads='default')                           
            model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', 
                                dataset_name, model_root_dirname)
        
    
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