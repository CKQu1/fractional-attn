import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
from time import sleep
from constants import *
from mutils import njoin, get_instance, structural_model_root
from qsub_parser import command_setup_ddp, qsub, job_divider
from qsub_parser import add_common_kwargs, get_pbs_array_data
from qsub_parser import job_setup

if __name__ == '__main__':
    
    # ----- System -----
    system = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS'    
    date_str = datetime.today().strftime('%Y-%m-%d')    
    script_name = "main.py"  
    nstack = 1    
    
    # ----- Paths -----
    ROOT = njoin(DROOT, 'full_model')
    job_path = njoin(ROOT, 'jobs_all', date_str)
    if not isdir(job_path): makedirs(job_path)

    #instances = list(range(5))    
    instances = [0]
    kwargss_all = []    

    DATASET_NAMES = ['imdb-classification', 'lra-cifar-classification',
                     'listops-classification', 'pathfinder-classification']  
    for instance in instances:        
        for didx, dataset_name in enumerate(DATASET_NAMES):
        #for didx, dataset_name in enumerate(DATASET_NAMES[0:1]):
            select = 1; ngpus, ncpus = 1, 0                            
            walltime, mem = '23:59:59', '32GB'                                         
            num_proc = ngpus if ngpus > 1 else ncpus
                        
            #for qk_share in [True, False]:
            for qk_share in [True]:
                kwargss = []

                for model_name in ['fnsformer', 'opfnsformer']:
                #for model_name in ['fnsformer']:
                    #for alpha in [1, 1.2, 1.4, 1.6, 1.8, 2]:
                    for alpha in [1.2, 1.6, 2]:
                        #for bandwidth in [0.01, 0.1, 0.5, 1]:
                        for bandwidth in [0.01, 0.1, 1]:
                        #for bandwidth in [0.1]:
                            kwargss.append({'model_name':model_name, 'alpha': alpha, 'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})

                kwargss.append({'model_name':'sinkformer', 'n_it': 1})
                kwargss.append({'model_name':'sinkformer', 'n_it': 3})
                kwargss.append({'model_name':'dpformer'})

                #epochs = DATASET_EPOCHS[dataset_name]
                epochs = None                                                              
                common_kwargs = {'instance':            instance,
                                'qk_share':             qk_share,
                                'num_encoder_layers':   1,
                                'num_heads':            1,                      
                                'train_bs':             16,   
                                'eval_bs':              16,                                                                       
                                'weight_decay':         0,
                                #'lr_scheduler_type':    'constant',
                                'lr_scheduler_type':    'cosine',
                                'max_lr':               1e-4,
                                'min_lr':               1e-5,
                                }  
                    
                common_kwargs['apples_to_apples'] = True
                common_kwargs['force_num_heads'] = True

                if not common_kwargs['apples_to_apples']:
                    if epochs is not None:                    
                        common_kwargs['epochs'] = epochs
                        #common_kwargs['eval_iters'] = 250
                    else:
                        common_kwargs['max_iters'] = 50
                        common_kwargs['eval_interval'] = 10
                        common_kwargs['eval_iters'] = 50                
                
                # if num_proc > 1:
                #     common_kwargs['grad_accum_step'] = num_proc * 2

                #use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                 
                model_root_dirname = structural_model_root(qk_share=qk_share, num_encoder_layers=common_kwargs['num_encoder_layers'],
                                                            num_attention_heads=common_kwargs['num_heads'] 
                                                            )  # hidden_size=common_kwargs['hidden_size']       

                #model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', model_root_dirname)
                model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', dataset_name, model_root_dirname)
                if not isdir(model_root): makedirs(model_root)                                                
                
                for idx in range(len(kwargss)):
                    # function automatically creates dir
                    kwargss[idx]["dataset"] = dataset_name    
                    kwargss[idx]['model_root'] = model_root
                
                kwargss = add_common_kwargs(kwargss, common_kwargs)
                kwargss_all += kwargss

    print(f'Total jobs: {len(kwargss_all)} \n')
    # for xx in kwargss_all:
    #     print(xx)  
    #     print('\n')
    #quit()  # delete     
     
    # ----- version 1 ----- 
    # train_submit(script_name, kwargss_all,
    #              ncpus=ncpus,
    #              ngpus=ngpus,
    #              select=select, 
    #              walltime=walltime,
    #              mem=mem,
    #              job_path=job_path,
    #              nstack=nstack,
    #              system=system)


    # ----- verion 2 -----

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
        qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], **kwargs_qsubs[i])      
