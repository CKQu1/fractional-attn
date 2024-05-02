import os
from os.path import isfile, isdir
from time import sleep
from constants import *
from mutils import njoin, get_instance
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

    # script for running
    script_name = "main.py" 
    # add or change datasets here
    dataset_names = ['rotten_tomatoes','imdb']
    max_lens = [256, 512]    
    
    debug_mode = False
    print(f'---------- debug_mode = {debug_mode} ---------- \n')
    
    kwargss_all = []
    #for didx, dataset_name in enumerate(dataset_names):
    for didx, dataset_name in enumerate(dataset_names[1:]):
        if not debug_mode:
            ngpus, ncpus = 0, 16
            #ngpus, ncpus = 0, 8
            select = 1
            walltime = '23:59:59'
            mem = '16GB'            

            #kwargss = [{'model_name': 'dpformer'}]                            
            #kwargss = [{'model_name': 'fnsformer', 'beta': 0.5, 'bandwidth': 1}]      
            #kwargss = [{'model_name': 'dpformer'}, {'model_name': 'spherefnsformer', 'beta': 0.5, 'bandwidth': 1}]                   
            # kwargss = [{'model_name': 'v2fnsformer', 'beta': 0.5, 'bandwidth': 5}, 
            #            {'model_name': 'v2fnsformer', 'beta': 0.5, 'bandwidth': 15}]
            #kwargss = [{'model_name': 'v2fnsformer', 'beta': 0.5, 'bandwidth': 5}, {'model_name': 'dpformer'}]
            # int(768/2)
            # kwargss = [{'model_name': 'v2fnsformer', 'beta': 1, 'bandwidth':15, 'd_intrinsic':10, 'qk_share':True}, 
            #            {'model_name': 'v2fnsformer', 'beta': 1, 'bandwidth':15, 'd_intrinsic':10, 'qk_share':False},
            #            {'model_name': 'v2fnsformer', 'beta': 2, 'bandwidth':15, 'qk_share':True},
            #            {'model_name': 'v2fnsformer', 'beta': 2, 'bandwidth':15, 'qk_share':False},
            #            {'model_name': 'dpformer', 'qk_share':True},
            #            {'model_name': 'dpformer', 'qk_share':False}]

            kwargss = [{'model_name':'dpformer'},
                       {'model_name':'v3fnsformer','beta':1.5},
                       {'model_name':'v3fnsformer','beta':2}]           
            model_root = njoin(DROOT, 'trained_models_v5')
            common_kwargs = {'n_layers':          1,
                             'n_attn_heads':      2,
                             'divider':           1,
                             'warmup_steps':      0, 
                             'grad_accum_step':   2,                            
                             'train_bs':          4,
                             'eval_bs':           4,
                             'max_len':           max_lens[didx],                             
                             'epochs':            10                                                   
                             }  

        else:     
                   
            ngpus, ncpus = 0, 2  
            select = 2  
            walltime = '23:59:59'
            mem = '12GB'                 

            #kwargss = [{'model_name': 'dpformer'}]
            #kwargss = [{"beta":0.5, "bandwidth":1}, {"beta":1, "bandwidth":1}]
            #kwargss = [{'model_name': 'dpformer'}, {'model_name': 'spherefnsformer', 'beta': 0.5, 'bandwidth': 1}]              
            kwargss = [{'model_name':'dpformer'},
                       {'model_name':'v3fnsformer','beta':1.5},
                       {'model_name':'v3fnsformer','beta':2}]              
            model_root = njoin(DROOT,'submit_main_check',f'ncpus={ncpus * select}_ngpus={ngpus * select}')
            common_kwargs = {'n_layers':                     1,
                             'n_attn_heads':                 2,
                             'divider':                      1,
                             'warmup_steps':                 0,                             
                             'train_bs':                     4,
                             'eval_bs':                      4,
                             'max_len':                      128,                             
                             'max_steps':                    50,
                             'logging_steps':                10,
                             'save_steps':                   10,
                             'eval_steps':                   10                         
                             }                                                     
        
        for idx in range(len(kwargss)):
            # function automatically creates dir
            kwargss[idx]["dataset"] = dataset_name    
            kwargss[idx]['model_root'] = model_root
        
        kwargss = add_common_kwargs(kwargss, common_kwargs)
        kwargss_all += kwargss

    #print(kwargss_all)  
    #quit()  # delete      
    train_submit(script_name, kwargss_all,
                 ncpus=ncpus,
                 ngpus=ngpus,
                 select=select, 
                 walltime=walltime,
                 mem=mem,
                 job_path=model_root)