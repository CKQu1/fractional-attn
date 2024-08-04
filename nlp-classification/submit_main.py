import json
import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
from time import sleep
from constants import *
from mutils import njoin, get_instance, structural_model_root, str2bool
from qsub_parser import command_setup, qsub, job_divider
from qsub_parser import command_setup_ddp

def add_common_kwargs(kwargss, common_kwargs):
    for i in range(len(kwargss)):
        for key, value in common_kwargs.items():
            kwargss[i][key] = value
    return kwargss

def get_pbs_array_data(kwargss):
    pbs_array_data = []
    for kwargs in kwargss:
        args_ls = []
        for key, value in kwargs.items():
            args_ls.append(f'--{key}={value}')
        pbs_array_data.append(tuple(args_ls))
    return pbs_array_data

# def train_submit(script_name, kwargss, **kwargs):

#     assert isfile(script_name), f"{script_name} does not exist!"

#     # computing resource settings
#     ncpus = kwargs.get('ncpus', 1) 
#     ngpus = kwargs.get('ngpus', 0)
#     walltime = kwargs.get('walltime', '23:59:59')
#     mem = kwargs.get('mem', '8GB')    
#     select = kwargs.get('select', 1)  # number of nodes    
#     command, additional_command = command_setup(SPATH,ncpus=ncpus,ngpus=ngpus,select=select)    
    
#     pbs_array_data = get_pbs_array_data(kwargss)    
    
#     perm, pbss = job_divider(pbs_array_data, len(PROJECTS))
#     for idx, pidx in enumerate(perm):
#         pbs_array_true = pbss[idx]
#         print(PROJECTS[pidx])
#         kwargs_qsub = {"path":     kwargs.get("job_path"),  # acts as PBSout
#                        "P":        PROJECTS[pidx],
#                        "ngpus":    ngpus, 
#                        "ncpus":    ncpus, 
#                        "select":   select,
#                        "walltime": walltime,
#                        "mem":      mem
#                        } 
#         if len(additional_command) > 0:
#             kwargs_qsub["additional_command"] = additional_command

#         qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
#         print("\n")

def train_submit(script_name, kwargss, **kwargs):

    assert isfile(script_name), f"{script_name} does not exist!"

    # system
    system = kwargs.get('system')
    nstack = kwargs.get('nstack', 1)

    # computing resource settings
    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)  # number of nodes    
    walltime = kwargs.get('walltime', '23:59:59')
    mem = kwargs.get('mem', '8GB')            
    
    pbs_array_data = get_pbs_array_data(kwargss)     
    if system == 'ARTEMIS':   
        perm, pbss = job_divider(pbs_array_data, len(PROJECTS))
    elif system == 'PHYSICS':
        perm, pbss = job_divider(pbs_array_data, 1)  # not needed for projects

    #master_port = 0
    HOST_NODE_ADDR = 0
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]        
        kwargs_qsub = {"path":        kwargs.get("job_path"),  # acts as PBSout                       
                       "ngpus":       ngpus, 
                       "ncpus":       ncpus, 
                       "select":      select,
                       "walltime":    walltime,
                       "mem":         mem,
                       "nstack":      nstack                       
                       }        

        kwargs_command = kwargs_qsub; del kwargs_command["path"]
        kwargs_command["system"] = system

        # ----- ARTEMIS -----
        if system == 'ARTEMIS':            
            # project names
            kwargs_qsub["P"] = PROJECTS[pidx]
            print(PROJECTS[pidx])

            if select * max(ncpus, ngpus) > 1:
                # master_port += 1            
                HOST_NODE_ADDR += 1

            kwargs_command["HOST_NODE_ADDR"] = HOST_NODE_ADDR
            kwargs_command["singularity_path"] = SPATH

        # ----- PHYSICS -----
        elif system == 'PHYSICS':
            if ngpus >= 1:
                kwargs_qsub["q"] = 'l40s'
            else:
                #kwargs_qsub["q"] = 'yossarian'
                pass

            kwargs_qsub["source"] = PHYSICS_SOURCE 
            kwargs_qsub["conda"] = PHYSICS_CONDA

        command, additional_command = command_setup_ddp(**kwargs_command)                

        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        #qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        qsub(f'{command} {script_name}', pbs_array_true, path=kwargs.get('job_path'), **kwargs_qsub)
        print("\n")        

if __name__ == '__main__':

    # ----- System -----
    system = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS'        
    date_str = datetime.today().strftime('%Y-%m-%d')    
    script_name = "main.py"
    nstack = 1  

    # settings
    config_files = ['config_qqv.json', 'config_qkv.json']  # 'config_qkv.json'
    # add or change datasets here
    DATASET_NAMES = ['rotten_tomatoes','imdb','emotion']
    #MAX_LENS = [128, 1024, 128]        
    MAX_LENS = [128, 512, 128]
    
    model_size = 'small'
    assert model_size in ['small', 'large']

    ROOT = njoin(DROOT, 'test-run')

    kwargss_all = []        
    for didx in [0]:
    #for didx in [1]:

        dataset_name = DATASET_NAMES[didx]
        
        select = 1; ngpus, ncpus = 0, 20  # CPUs        
        #select = 1; ngpus, ncpus = 1, 0  # GPUs

        mem = '20GB'       
        #mem = '12GB'  # 1L1H
        #mem = '32GB'  # 6L8H (512 len)
                     
        walltime = '23:59:59'            

        seeds = [0]   
                                                  
        #for config_file in config_files:        
        for qk_share in [False]:

            kwargss = []
            # for alpha in [1.2, 1.6, 2]:
            # #for alpha in [1.2, 2]:
            #     #for bandwidth in [0.01, 0.1, 0.5, 1]:                                    
            #     for bandwidth in [1]:
            #         kwargss.append({'model_name':'opfnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})
            #         #kwargss.append({'model_name':'opfnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'rd'})
            #         #kwargss.append({'model_name':'fnsformer','alpha':alpha,'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})

            kwargss.append({'model_name': 'dpformer'})
            for n_it in [3]:
                kwargss.append({'model_name': 'sinkformer', 'n_it': n_it})

            # f = open(njoin('models', 'all_configs', config_file))
            # common_kwargs = json.load(f)
            # f.close()        

            common_kwargs = {                                     
                "qk_share":          qk_share,
                "hidden_size":       768,
                "warmup_steps":      0, 
                "grad_accum_step":   2,                            
                "train_bs":          32,
                "eval_bs":           32,                                            
                "lr_scheduler_type": "constant",
                "lr":                1e-4,    
                #"gamma":             0.1,      
                "gamma":             0,
                "milestones":        "",                                          
                "weight_decay":      0
            }  

            if model_size == 'large':
                common_kwargs["n_layers"] = 6
                common_kwargs["n_attn_heads"] = 8
                common_kwargs["epochs"] = 15
            elif model_size == 'small':
                common_kwargs["n_layers"] = 1
                common_kwargs["n_attn_heads"] = 1                             

            # add more settings here
            common_kwargs['max_len'] = MAX_LENS[didx]            
            common_kwargs['fix_embed'] = False

            # test-run
            common_kwargs['max_steps'] = 10
            common_kwargs['logging_steps'] = 5
            common_kwargs['eval_steps'] = 5
            common_kwargs['save_steps'] = 5

            for seed in seeds:                                                             

                common_kwargs['seed'] = seed                
                #use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                                        
                model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)      
                
                job_path = njoin(ROOT, 'jobs_all', date_str)
                #model_root = njoin(ROOT, config_file.split('.')[0], model_root_dirname)
                model_root = njoin(ROOT, 'config_qqv' if qk_share else 'config_qkv', model_root_dirname)

                if not isdir(model_root): makedirs(model_root)
                if not isdir(job_path): makedirs(job_path)

                for idx in range(len(kwargss)):                    
                    kwargss[idx]["dataset"] = dataset_name    
                    kwargss[idx]['model_root'] = model_root
                
                kwargss = add_common_kwargs(kwargss, common_kwargs)
                kwargss_all += kwargss                

    print(f'Total jobs: {len(kwargss_all)} \n')
    # for xx in kwargss_all:
    #     print(xx)  
    #     print('\n')     
    #quit()
    train_submit(script_name, kwargss_all,
                 ncpus=ncpus,
                 ngpus=ngpus,
                 select=select, 
                 walltime=walltime,
                 mem=mem,
                 job_path=job_path,
                 nstack=nstack,
                 system=system)