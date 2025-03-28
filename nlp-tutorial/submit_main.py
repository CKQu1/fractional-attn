import json
import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
from time import sleep
from constants import DROOT, CLUSTER
from utils.mutils import njoin, get_instance, structural_model_root, str2bool
#from qsub_parser import *
from qsub_parser import job_setup, qsub, add_common_kwargs

if __name__ == '__main__':
       
    date_str = datetime.today().strftime('%Y-%m-%d')    
    script_name = "main.py"
    nstack = 1  

    # settings
    # add or change datasets here
    DATASET_NAMES = ['rotten_tomatoes','imdb','emotion']     
    MAX_LENS = [128, 512, 128]

    #instances = [0]
    instances = list(range(5))
    #model_sizes = ['small']
    model_sizes = ['medium']
    #model_sizes = ['large']

    hidden = 256
    #ROOT = njoin(DROOT, f'full-model-d={hidden}')
    #ROOT = njoin(DROOT, f'full-model-v2')
    ROOT = njoin(DROOT, f'medium-model-v2')
    #ROOT = njoin(DROOT, f'full-model-v3')
    #ROOT = njoin(DROOT, f'debug-mode')
    job_path = njoin(ROOT, 'jobs_all')

    for model_size in model_sizes:
        kwargss_all = []    
        for instance in instances:
            #for didx, dataset_name in enumerate(DATASET_NAMES):   
            for didx in [1]:
            #for didx in [0]:

                dataset_name = DATASET_NAMES[didx]
                
                # CPUs
                #select = 1; ngpus, ncpus = 0, 8; mem = '24GB'  # imdb          
                #select = 1; ngpus, ncpus = 0, 16; mem = '32GB'  # rotten_tomatoes
                # GPUs
                if model_size == 'large':
                    select = 1; ngpus, ncpus = 1, 1; mem = '10GB'  # 6L8H imdb (512 len)  
                elif model_size == 'medium':
                    select = 1; ngpus, ncpus = 1, 1; mem = '8GB'  # 6L8H imdb (512 len)                      
                else:
                    select = 1; ngpus, ncpus = 1, 1; mem = '6GB'  # 1L1H imdb (512 len)                                   
                                
                walltime = '23:59:59'                    
                train_with_ddp = max(ncpus, ngpus) > 1


                for qk_share in [True, False]:
                #for qk_share in [True]:

                    #for is_op in [True, False]:
                    for is_op in [True]:

                        kwargss = []

                        for alpha in [1.2, 2]:
                            for bandwidth in [1]:
                                for manifold in ['rd']: 
                                #for manifold in ['rd', 'sphere']:
                                    kwargss.append({'model_name':'fnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':manifold,
                                                    'is_op': is_op})

                        kwargss.append({'model_name':'dpformer','is_op': is_op})

                        for n_it in [3]:
                            kwargss.append({'model_name':'sinkformer','n_it':n_it,'is_op': is_op})

                        common_kwargs = {        
                            "instance":          instance,                             
                            "seed":              instance,
                            "qk_share":          qk_share,
                            #"hidden":           768,
                            "hidden":            hidden,
                            #"hidden":           32,                          
                            "train_bs":          32,                                          
                            #"lr_scheduler_type": "constant",                           
                            "lr_scheduler_type": "binary",                                       
                            "weight_decay":      0,
                            'max_lr':            1e-4,
                            'min_lr':            1e-5                            
                        }  

                        #common_kwargs['n_attn_heads'] = 1 if is_op else 8
                        common_kwargs['n_attn_heads'] = 8
     
                        if model_size == 'large':
                            common_kwargs["n_layers"] = 6
                            #common_kwargs["n_attn_heads"] = 8
                            #common_kwargs["epochs"] = 15
                            common_kwargs["epochs"] = 20
                            #common_kwargs['fix_embed'] = False

                            common_kwargs["max_lr"] = 1e-4

                        elif model_size == 'medium':
                            common_kwargs["n_layers"] = 4
                            common_kwargs["epochs"] = 20

                        elif model_size == 'small':
                            common_kwargs["n_layers"] = 2
                            #common_kwargs["n_attn_heads"] = 1                             
                            #common_kwargs["epochs"] = 15
                            common_kwargs["epochs"] = 1
                            #common_kwargs['fix_embed'] = True
                            #common_kwargs['fix_embed'] = False

                            #common_kwargs["lr"] = 1e-4
                            common_kwargs["max_lr"] = 5e-4

                            #common_kwargs["hidden"] = 12
                            #common_kwargs["hidden"] = 48
                            common_kwargs["hidden"] = 64

                        # add more settings here
                        common_kwargs['max_len'] = MAX_LENS[didx]                    

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
        
        # for i in range(len(commands)):
        #     qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])          