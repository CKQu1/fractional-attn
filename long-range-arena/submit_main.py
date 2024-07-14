import os
from datetime import datetime
from os import makedirs
from os.path import isdir, isfile
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

if __name__ == '__main__':
    
    # datetime
    date_str = datetime.today().strftime('%Y-%m-%d')    
    script_name = "main.py"
    dataset_names = ['imdb-classification', 'lra-cifar-classification',
                     'listops-classification', 'aan-classification',
                     'pathfinder-classification', 'pathx-classification']   
    
    dataset_epochs = {'imdb-classification':5, 'lra-cifar-classification':100,
                      'listops-classification':5, 'aan-classification':5,
                      'pathfinder-classification':5, 'pathx-classification':5}
    
    # ----- Paths -----
    ROOT = njoin(DROOT, 'trained_models')
    job_path = njoin(ROOT, 'jobs_all', date_str)
    if not isdir(job_path): makedirs(job_path)

    #instances = list(range(5))    
    instances = [0]
    kwargss_all = []    
    for instance in instances:        
        #for didx, dataset_name in enumerate(dataset_names):
        for didx, dataset_name in enumerate(dataset_names[1:3]):
        #for didx, dataset_name in enumerate(dataset_names[2:3]):
            select = 1; ngpus, ncpus = 1, 0                            
            walltime, mem = '23:59:59', '8GB'                                         
            num_proc = ngpus if ngpus > 1 else ncpus
                        
            for qk_share in [True, False]:
                kwargss = []

                for model_name in ['fnsformer', 'opfnsformer']:
                    for alpha in [1.2, 1.6, 2]:
                    #for alpha in [1.2, 2]:
                        for bandwidth in [0.01, 0.1, 0.5, 1]:
                        #for bandwidth in [0.1]:
                            kwargss.append({'model_name':model_name, 'alpha': alpha, 'a': 0,'bandwidth':bandwidth,'manifold':'sphere'})

                kwargss.append({'model_name':'sinkformer', 'n_it': 1})
                kwargss.append({'model_name':'sinkformer', 'n_it': 3})
                kwargss.append({'model_name':'dpformer'})

                epochs = dataset_epochs[dataset_name]
                #epochs = None                                                              
                common_kwargs = {'instance':            instance,
                                'qk_share':             qk_share,
                                'num_encoder_layers':   1,
                                'num_heads':            1,                      
                                'train_bs':             16,   
                                'eval_bs':              16,                                                                       
                                'weight_decay':         0,
                                'lr_scheduler_type':    'constant',
                                'max_lr':               5e-5,
                                'min_lr':               5e-6,
                                }  

                if epochs is not None:                    
                    common_kwargs['epochs'] = epochs
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
    train_submit(script_name, kwargss_all,
                 ncpus=ncpus,
                 ngpus=ngpus,
                 select=select, 
                 walltime=walltime,
                 mem=mem,
                 job_path=job_path)