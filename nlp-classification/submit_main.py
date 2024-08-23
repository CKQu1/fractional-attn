import json
import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
from time import sleep
from constants import DROOT, CLUSTER
from mutils import njoin, get_instance, structural_model_root, str2bool
#from qsub_parser import *
from qsub_parser import job_setup, qsub, add_common_kwargs

if __name__ == '__main__':
       
    date_str = datetime.today().strftime('%Y-%m-%d')    
    script_name = "main.py"
    nstack = 1  

    # settings
    config_files = ['config_qqv.json', 'config_qkv.json']  # 'config_qkv.json'
    # add or change datasets here
    DATASET_NAMES = ['rotten_tomatoes','imdb','emotion']
    #MAX_LENS = [128, 1024, 128]        
    MAX_LENS = [128, 512, 128]

    instances = [0]
    model_sizes = ['small']
    #model_sizes = ['large']

    #ROOT = njoin(DROOT, 'new-fix-embed')
    ROOT = njoin(DROOT, 'rt-test-v3')
    #job_path = njoin(ROOT, 'jobs_all', date_str)
    job_path = njoin(ROOT, 'jobs_all')

    for model_size in model_sizes:
        kwargss_all = []    
        for instance in instances:
            #for didx, dataset_name in enumerate(DATASET_NAMES):   
            #for didx in [1]:
            for didx in [0]:

                dataset_name = DATASET_NAMES[didx]
                
                # CPUs
                #select = 1; ngpus, ncpus = 0, 8; mem = '24GB'  # imdb          
                select = 1; ngpus, ncpus = 0, 16; mem = '32GB'  # rotten_tomatoes
                # GPUs
                #select = 1; ngpus, ncpus = 1, 0; #mem = '32GB'  # 6L8H (512 len)                                     
                                
                walltime = '23:59:59'                    
                train_with_ddp = max(ncpus, ngpus) > 1

                #for qk_share in [True, False]:
                for qk_share in [True]:

                    kwargss = []
                    for alpha in [1.2, 2]:        
                    #for alpha in [1, 1.4, 1.8]:                    
                        #for bandwidth in [0.01, 0.1, 0.5, 1]:                                    
                        for bandwidth in [0.1,1]:
                        #for bandwidth in [1]:
                            kwargss.append({'model_name':'opfnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})
                            kwargss.append({'model_name':'opfnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'rd'})
                            #kwargss.append({'model_name':'fnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})                        

                            #kwargss.append({'model_name':'opfnsformer','manifold':'sphere','alpha': 1.2,'a': 0,'bandwidth':1})

                    kwargss.append({'model_name':'dpformer'})
                    for n_it in [3]:
                        kwargss.append({'model_name':'sinkformer','n_it':n_it})
                        
                    common_kwargs = {        
                        "instance":          instance,                             
                        "seed":              instance,
                        "qk_share":          qk_share,
                        "hidden_size":       768,
                        "warmup_steps":      0, 
                        "grad_accum_step":   2,                            
                        "train_bs":          32,
                        "eval_bs":           32,                                            
                        "lr_scheduler_type": "constant",                           
                        #"gamma":             0.1,      
                        "gamma":             0,
                        "milestones":        "",                                          
                        "weight_decay":      0,
                        "train_with_ddp":    train_with_ddp
                    }  

                    if model_size == 'large':
                        common_kwargs["n_layers"] = 6
                        common_kwargs["n_attn_heads"] = 8
                        common_kwargs["epochs"] = 15
                        common_kwargs['fix_embed'] = False

                        common_kwargs["lr"] = 1e-4

                    elif model_size == 'small':
                        common_kwargs["n_layers"] = 1
                        common_kwargs["n_attn_heads"] = 1                             
                        common_kwargs["epochs"] = 15
                        #common_kwargs['fix_embed'] = True
                        common_kwargs['fix_embed'] = False

                        common_kwargs["lr"] = 1e-4

                    # add more settings here
                    common_kwargs['max_len'] = MAX_LENS[didx]                    

                    # test-run
                    # common_kwargs["train_bs"] = 4; common_kwargs["eval_bs"] = 4
                    # common_kwargs['max_steps'] = 10
                    # common_kwargs['logging_steps'] = 5
                    # common_kwargs['eval_steps'] = 5
                    # common_kwargs['save_steps'] = 5
                    # common_kwargs["epochs"] = 1

                    #use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                                        
                    model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                          
                    #model_root = njoin(ROOT, config_file.split('.')[0], dataset_name, model_root_dirname)
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
                        cluster=CLUSTER)
        
        for i in range(len(commands)):
            qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])          