import os
from os.path import isfile, join
from path_setup import droot
repo_dir = os.getcwd()  # main dir

# for enumerating each instance of training
def get_instance(dir, s):
    instances = []
    dirnames = next(os.walk(dir))[1]
    if len(dirnames) > 0:
        for dirname in dirnames:        
            if s in dirname and len(os.listdir(join(dir, dirname))) > 0:
                try:                
                    instances.append(int(dirname.split(s)[-1]))
                except:
                    pass        
        return max(instances) + 1 if len(instances)>0 else 0
    else:
        return 0

def create_model_dir(model_root_dir, **kwargs):
    with_frac = kwargs.get("with_frac", False)
    gamma = kwargs.get('gamma', None)
    dataset_name = kwargs.get('dataset_name', 'imdb')
    model_name = kwargs.get('model_name', 'diffuser')
    if (model_name == 'diffuser') and with_frac:                   
        model_root_dir = join(model_root_dir, f"save_{dataset_name}_frac_{model_name}")
    else:            
        model_root_dir = join(model_root_dir, f"save_{dataset_name}_{model_name}")
    if not os.path.isdir(model_root_dir): os.makedirs(model_root_dir)    
    if with_frac:    
        instance = get_instance(model_root_dir, f"gamma={gamma}")
        model_dir = join(model_root_dir, f"model={instance}_gamma={gamma}")
    else:
        instance = get_instance(model_root_dir, "model=")
        model_dir = join(model_root_dir, f"model={instance}")     
    if not os.path.isdir(model_dir): os.makedirs(model_dir)        
    return model_dir    

def command_setup(ngpus, ncpus, singularity_path):
    assert isfile(singularity_path), "singularity_path does not exist!"

    if len(singularity_path) > 0:
        command = f"singularity exec --home {repo_dir} {singularity_path}"
    else:
        command = ""

    additional_command = ''
    train_with_ddp = False
    if max(ngpus, ncpus) <= 1:
        command += f" python"
    elif ngpus > 1:
        #command += f" CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node={ngpus}"
        #additional_command = 'run --backend=nccl'
        command += f" torchrun --nproc_per_node={ngpus}"        
        train_with_ddp = True
    elif ngpus == 0 and ncpus > 1:
        #python -m torch.distributed.launch --nproc_per_node=4 --use_env train_classification_imdb.py run --backend=gloo
        #additional_command = 'run --backend=gloo'
        command += f" torchrun --nproc_per_node={ncpus}"
        train_with_ddp = True        

    if len(singularity_path) == 0:
        command = command[1:]

    return command, additional_command, train_with_ddp

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

def train_submit(script_name, ngpus, ncpus, kwargss, **kwargs):
    assert isfile(script_name), f"{script_name} does not exist!"

    # computing resource settings
    ngpus, ncpus = int(ngpus), int(ncpus)
    singularity_path = "../built_containers/FaContainer_v2.sif"
    command, additional_command, train_with_ddp = command_setup(ngpus, ncpus, singularity_path)    

    from qsub_parser import qsub, job_divider
    project_ls = ["ddl"]  # can add more projects here
    pbs_array_data = get_pbs_array_data(kwargss)
    #print(pbs_array_data)    
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        kwargs_qsub = {"path":kwargs.get("job_path"),  # acts as PBSout
                       "P":project_ls[pidx],
                       "ngpus":ngpus, 
                       "ncpus":ncpus, 
                       "walltime":'23:59:59', 
                      #"walltime":'0:29:59', 
                       "mem":"16GB"}        
        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        print("\n")

if __name__ == '__main__':

    # script for running
    script_name = "main_seq_classification.py"
    ngpus, ncpus = 0, 4
    train_with_ddp = True if max(ngpus, ncpus) > 1 else False    
    
    debug_mode = True
    if not debug_mode:
        kwargss = [{}, {"with_frac":True, "gamma":0.25}, 
                   {"with_frac":True, "gamma":0.5}, {"with_frac":True, "gamma":0.75}]  # empty dict is diffuser

        model_root_dir = join(droot, "main_seq_classification")
        common_kwargs = {"gradient_accumulation_steps":4,
                         #"epochs": 0.1,
                         "max_steps": 50,
                         "warmup_steps":10, "eval_steps":5, "logging_steps":5, "save_steps":5,
                         "per_device_eval_batch_size":2}
    else:
        kwargss = [{}, {"with_frac":True, "gamma":0.25}, {"with_frac":True, "gamma":0.5}, 
                   {"with_frac":True, "gamma":0.75} ]  # empty dict is diffuser
        model_root_dir = join(droot, "debug_mode6")
        common_kwargs = {"gradient_accumulation_steps":2,
                         "max_steps": 2,
                         "warmup_steps":0, "eval_steps":2, "logging_steps":2, "save_steps":2,
                         "per_device_train_batch_size":2,
                         "per_device_eval_batch_size":2} 
    if train_with_ddp:
        common_kwargs["train_with_ddp"] = train_with_ddp
    kwargss = add_common_kwargs(kwargss, common_kwargs)
    
    for idx in range(len(kwargss)):
        # function automatically creates dir
        kwargss[idx]["model_dir"] = create_model_dir(model_root_dir, **kwargss[idx])
    #print(kwargss)
    train_submit(script_name, ngpus, ncpus, kwargss, job_path=model_root_dir)