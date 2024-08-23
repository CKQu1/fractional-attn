import json
import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
from time import sleep
from constants import DROOT, CLUSTER
from mutils import njoin, get_instance, structural_model_root, str2bool
from mutils import create_model_dir
#from qsub_parser import *
from qsub_parser import job_setup, qsub, add_common_kwargs

if __name__ == '__main__':
       
    script_name = "main.py"
    nstack = 1  

    # settings
    # TASKS = ["lra-listops", "lra-retrieval", "lra-text",  
    #          "lra-image", "lra-pathfinder32-curv_contour_length_14"]
    TASKS = ["lra-listops", "lra-pathfinder32-curv_contour_length_14"]
    #TASKS = ["lra-retrieval", "lra-text", "lra-image"]             
    MEM_DICT = {"lra-listops": "12GB", "lra-retrieval": "32GB", "lra-text": "32GB",  
                "lra-image": "32GB", "lra-pathfinder32-curv_contour_length_14": "12GB"}
    seeds = [0]

    #ROOT = njoin(DROOT, 'V100-run')
    ROOT = njoin(DROOT, 'V100-run-cycliclr')
    job_path = njoin(ROOT, 'jobs_all')

    kwargss_all = []    
    for seed in seeds:
        for task in TASKS:
            
            # CPUs
            #select = 1; ngpus, ncpus = 0, 1; # mem = '24GB' 
            # GPUs
            #select = 1; ngpus, ncpus = 1, 0; mem = '32GB'  # 6L8H (512 len)                                     
            select = 1; ngpus, ncpus = 1, 1; # mem = '24GB'  #2L2H
            mem = MEM_DICT[task]
                            
            walltime = '23:59:59'                    
            train_with_ddp = max(ncpus, ngpus) > 1

            #for qk_share in [True, False]:

            kwargss = []
            for alpha in [1.2,2]:        
            #for alpha in [1, 1.4, 1.8]:                                                       
                for bandwidth in [0.01, 1]:
                    kwargss.append({'attn':'opfns','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})
                    #kwargss.append({'attn':'opfns','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'rd'})

            #if task == 'lra-text':
            kwargss.append({'attn':'softmax'})
            
            # for n_it in [3]:
            #     kwargss.append({'attn':'sink','n_it':n_it})

            common_kwargs = {      
                "lr_scheduler":    'cycliclr',      
                "task":            task,                         
                "random":          seed,
                "qkv_bias":        False
            } 
            
            kwargss = add_common_kwargs(kwargss, common_kwargs)

            #model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)                                   
            #log_dir = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', task, model_root_dirname)            
                
            for idx in range(len(kwargss)):
                _, log_dir = create_model_dir(ROOT, **kwargss[idx])
                kwargss[idx]['log_dir'] = log_dir
                        
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
            qsub(f'{commands[i]} {script_names[i]}', pbs_array_trues[i], **kwargs_qsubs[i])          