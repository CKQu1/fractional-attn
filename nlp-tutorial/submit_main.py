import os
from os.path import isfile
from time import sleep
from constants import DROOT, BPATH, SPATH, PROJECTS
from path_names import njoin, get_instance
from qsub_parser import command_setup, qsub, job_divider

def create_model_dir(model_root_dir, **kwargs):
    beta = kwargs.get("beta", 1)
    bandwidth = kwargs.get("bandwidth", 1)
    dataset_name = kwargs.get('dataset_name', 'imdb')
    model_name = kwargs.get('model_name', 'fnsformer')
          
    models_dir = njoin(model_root_dir, f"{model_name}_{dataset_name}")
    if not os.path.isdir(models_dir): os.makedirs(models_dir)    
    instance = get_instance(models_dir, f"_beta={beta}_eps={bandwidth}")
    model_dir = njoin(models_dir, f"model={instance}_beta={beta}_eps={bandwidth}")    
    if not os.path.isdir(model_dir): os.makedirs(model_dir)     
       
    return models_dir, model_dir    

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
    script_name = 'main.py'
    # add or change datasets here
    dataset_names = ['imdb']  # 'rotten_tomatoes'    
    
    #ncpuss = [2]
    ncpuss = [1,4,8,12]
    #ncpuss = list(range(1,9))
    ngpus = 0
    select = 1
    walltime = '23:59:59'
    mem = '12GB'

    debug_mode = True
    print(f'debug_mode = {debug_mode}')

    for ii, ncpus in enumerate(ncpuss):
        for dataset_name in dataset_names:
            if not debug_mode:
                ngpus, ncpus = 0, 12
                select = 1
                                
                kwargss = [{}, {"with_frac":True, "gamma":0.4}]                         
                model_root_dir = njoin(DROOT, 'trained_models', "l2_mha")  # L2-MHA (without AX)
                common_kwargs = {"gradient_accumulation_steps":4,
                                #"epochs":20,
                                "epochs":10,
                                "warmup_steps":50,
                                "divider": 1,
                                "per_device_train_batch_size":4,
                                "per_device_eval_batch_size":4
                                }
            else:            
                #ngpus, ncpus = 0, 2
                
                kwargss = [{"beta":0.5, "bandwidth":1}, {"beta":1, "bandwidth":1}]
                #kwargss = [{"beta":0.75, "bandwidth":1}, {"beta":1, "bandwidth":1}]
                #model_root_dir = njoin(DROOT, "qsub_check")
                model_root_dir = njoin(DROOT,'dp_check',f'ncpus={ncpus}')

                common_kwargs = {'epochs':         10,                             
                                 'n_attn_heads':   1,
                                 'batch_size':     8,
                                 'max_seq_len':    200,
                                 'divider':        100,
                                 'n_layers':       2,
                                 'n_attn_heads':   1
                                 }                            
            
            for idx in range(len(kwargss)):
                # function automatically creates dir
                kwargss[idx]["dataset"] = dataset_name
                #models_dir, kwargss[idx]["model_dir"] = create_model_dir(model_root_dir, **kwargss[idx])         
                kwargss[idx]['models_dir'] = model_root_dir
            
            kwargss = add_common_kwargs(kwargss, common_kwargs)
            #print(kwargss)        
            train_submit(script_name, kwargss,
                         ncpus=ncpus,
                         ngpus=ngpus,
                         select=select, 
                         walltime=walltime,
                         mem=mem,
                         job_path=model_root_dir)