import os
from os.path import isfile, isdir
from time import sleep
from constants import *
from mutils import njoin, get_instance, structural_model_root
from qsub_parser import command_setup_ddp, qsub, job_divider

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
    select = kwargs.get('select', 1)  # number of nodes    

    walltime = kwargs.get('walltime', '23:59:59')
    mem = kwargs.get('mem', '8GB')            
    
    pbs_array_data = get_pbs_array_data(kwargss)        
    perm, pbss = job_divider(pbs_array_data, len(PROJECTS))

    #master_port = 0
    HOST_NODE_ADDR = 0
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(PROJECTS[pidx])
        kwargs_qsub = {"path":        kwargs.get("job_path"),  # acts as PBSout
                       "P":           PROJECTS[pidx],
                       "ngpus":       ngpus, 
                       "ncpus":       ncpus, 
                       "select":      select,
                       "walltime":    walltime,
                       "mem":         mem                       
                       }        

        if select * max(ncpus, ngpus) > 1:
            # master_port += 1            
            HOST_NODE_ADDR += 1

        command, additional_command = command_setup_ddp(SPATH,ncpus=ncpus,ngpus=ngpus,select=select,
                                                        HOST_NODE_ADDR=HOST_NODE_ADDR)                           

        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        print("\n")

"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
    
    script_name = "ddp_main.py"       # script for running
    dataset_names = ['cifar10']   # add or change datasets here
    
    debug_mode = False
    print(f'---------- debug_mode = {debug_mode} ---------- \n')
    
    seeds = [0]    
    kwargss_all = []    
    for seed in seeds:
        for didx, dataset_name in enumerate(dataset_names):
            if not debug_mode:
                select = 1; ngpus, ncpus = 0, 1            
                walltime, mem = '23:59:59', '16GB'                             

                kwargss = [{'model_name':'fnsvit', 'beta': 1.5}, {'model_name':'fnsvit', 'beta': 2}, 
                           {'model_name':'dpvit'}]                                        
                common_kwargs = {'n_layers':          5,
                                 'n_attn_heads':      8,   
                                 'hidden_size':       48,
                                 'max_iters':         2500,
                                 'eval_interval':     50,
                                 'eval_iters':        50,                     
                                 'train_bs':          16,                                                                          
                                 'weight_decay':      0,
                                 'max_lr':            5e-5,
                                 'min_lr':            1e-6,
                                 }  

                qk_share = False if 'qk_share' not in common_kwargs.keys() else common_kwargs['qk_share']
                use_custom_optim = False if 'use_custom_optim' not in common_kwargs.keys() else common_kwargs['use_custom_optim']                                 

                model_root_dirname = structural_model_root(qk_share=qk_share, n_layers=common_kwargs['n_layers'],
                                                           n_attn_heads=common_kwargs['n_attn_heads'], hidden_size=common_kwargs['hidden_size']
                                                           )       
                model_root = njoin(DROOT, 'formers_trained', model_root_dirname)

            else:                         
                ngpus, ncpus = 0, 8
                select = 1  
                walltime, mem = '23:59:59', '8GB'                
        
                kwargss = [{'model_name':'fnsvit', 'beta': 1.5}, {'model_name':'fnsvit', 'beta': 2}, 
                           {'model_name':'dpvit'}]            
                common_kwargs = {'n_layers':          2,
                                 'n_attn_heads':      2,    
                                 'max_iters':         1000,
                                 'eval_interval':     200,
                                 'eval_iters':        200,                     
                                 'train_bs':          8,                                                                          
                                 'weight_decay':      0
                                 }      

                #model_root = njoin(DROOT, 'ddp_test_stage')                                                                                
                model_root = njoin(DROOT, f'select={select}-ncpus={ncpus}-ngpus={ngpus}')
            
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