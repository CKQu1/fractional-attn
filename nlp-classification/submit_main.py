import json
import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
from time import sleep
from constants import *
from mutils import njoin, get_instance, structural_model_root, str2bool
from qsub_parser import command_setup, qsub, job_divider

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

def train_submit(script_name, kwargss, **kwargs):

    assert isfile(script_name), f"{script_name} does not exist!"

    # computing resource settings
    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    walltime = kwargs.get('walltime', '23:59:59')
    mem = kwargs.get('mem', '8GB')    
    select = kwargs.get('select', 1)  # number of nodes    
    command, additional_command = command_setup(SPATH,ncpus=ncpus,ngpus=ngpus,select=select)    
    
    pbs_array_data = get_pbs_array_data(kwargss)    
    
    perm, pbss = job_divider(pbs_array_data, len(PROJECTS))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(PROJECTS[pidx])
        kwargs_qsub = {"path":     kwargs.get("job_path"),  # acts as PBSout
                       "P":        PROJECTS[pidx],
                       "ngpus":    ngpus, 
                       "ncpus":    ncpus, 
                       "select":   select,
                       "walltime": walltime,
                       "mem":      mem
                       } 
        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        print("\n")

if __name__ == '__main__':

    # datetime
    date_str = datetime.today().strftime('%Y-%m-%d')
    # script for running
    script_name = "main.py" 
    # settings
    config_files = ['config_qqv.json', 'config_qkv.json']  # 'config_qkv.json'
    # add or change datasets here
    dataset_names = ['rotten_tomatoes','imdb','emotion']
    max_lens = [128, 1024, 128]        
    
    kwargss_all = []        
    #for didx in [0,1]:
    #for didx in [0]:
    for didx in [1]:

        dataset_name = dataset_names[didx]

        # CPUs
        select = 1; ngpus, ncpus = 0, 20
        #select = 2; ngpus, ncpus = 0, 12 
        mem = '20GB'       
         
        # GPUs
        # select = 1; ngpus, ncpus = 1, 0 
        # mem = '16GB'
            
        walltime = '23:59:59'            

        seeds = [0]                                             
        for config_file in config_files:
            # kwargss = [{'model_name':'opfnsformer','alpha':1.6,'a': 0,'bandwidth':1,'manifold':'sphere'},   
            #            {'model_name':'opfnsformer','alpha':1.8,'a': 0,'bandwidth':1,'manifold':'sphere'},                   
            #            {'model_name':'opfnsformer','alpha':2,'a': 0,'bandwidth':1,'manifold':'sphere'}                                      
            #            ] 

            kwargss = [{'model_name':'fnsformer','alpha':2,'a': 0,'bandwidth':0.1,'manifold':'sphere'},                  
                       {'model_name':'opfnsformer','alpha':2,'a': 0,'bandwidth':0.1,'manifold':'sphere'}                                      
                       ] 

            # {'model_name':'fnsformer','alpha':1.5,'a': 0,'bandwidth':1,'manifold':'rd'},                      
            # {'model_name':'fnsformer','alpha':2,'a': 0,'bandwidth':1,'manifold':'rd'}                        

            f = open(njoin('models', 'all_configs', config_file))
            common_kwargs = json.load(f)
            f.close()        
            common_kwargs['max_len'] = max_lens[didx]
            common_kwargs['qk_share'] = str2bool(common_kwargs['qk_share'])

            for seed in seeds:                                                             

                common_kwargs['seed'] = seed                
                #use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                                        
                model_root_dirname = structural_model_root(dataset_name=dataset_name, **common_kwargs)      

                #model_root = njoin(DROOT, 'finetune-v5', model_root_dirname)                
                #model_root = njoin(DROOT, config_file.split('.')[0], model_root_dirname)
                ROOT = njoin(DROOT, 'trained_models_v2')
                job_path = njoin(ROOT, date_str)
                model_root = njoin(ROOT, config_file.split('.')[0], model_root_dirname)
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
    train_submit(script_name, kwargss_all,
                 ncpus=ncpus,
                 ngpus=ngpus,
                 select=select, 
                 walltime=walltime,
                 mem=mem,
                 job_path=ROOT)