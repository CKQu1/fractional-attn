import os
from os.path import isfile, join
from time import sleep
from path_setup import droot
repo_dir = os.getcwd()  # main dir

# for enumerating each instance of training
def get_instance(dir, s):
    instances = []
    dirnames = next(os.walk(dir))[1]
    if len(dirnames) > 0:
        for dirname in dirnames:        
            # len(os.listdir(join(dir, dirname))) > 0
            if s in dirname and "model=" in dirname:
                #try:        
                for s_part in dirname.split(s):
                    if "model=" in s_part:
                        start = s_part.find("model=") + 6
                        end = s_part.find("_")
                        instances.append(int(s_part[start:end]))
                #except:
                #    pass       
        print(instances) 
        return max(instances) + 1 if len(instances)>0 else 0
    else:
        return 0

def create_model_dir(model_root_dir, **kwargs):
    with_frac = kwargs.get("with_frac", False)
    gamma = kwargs.get('gamma', None)
    dataset_name = kwargs.get('dataset_name', 'imdb')
    model_name = kwargs.get('model_name', 'diffuser')
    if (model_name == 'diffuser') and with_frac:                   
        models_dir = join(model_root_dir, f"frac_{model_name}")
    else:            
        models_dir = join(model_root_dir, f"{model_name}")
    if not os.path.isdir(models_dir): os.makedirs(models_dir)    
    if with_frac:    
        instance = get_instance(models_dir, f"gamma={gamma}")
        model_dir = join(models_dir, f"model={instance}_gamma={gamma}")
    else:
        instance = get_instance(models_dir, "model=")
        model_dir = join(models_dir, f"model={instance}")     
    if not os.path.isdir(model_dir): os.makedirs(model_dir)        
    return models_dir, model_dir    

def command_setup(ngpus, ncpus, singularity_path, **kwargs):
    assert isfile(singularity_path), "singularity_path does not exist!"

    select = kwargs.get('select', 1)
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
        if select == 1:
            command += f" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0"
            command += f" --nnodes=1 --nproc_per_node={ngpus} --max-restarts=3"
        else:
            # tolerates 3 failures
            command += f" torchrun --nnodes={select} --nproc_per_node={ngpus}"
            command += f" --max-restarts=3 --rdzv-id=$JOB_ID"
            command += f" --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"
        train_with_ddp = True
    elif ngpus == 0 and ncpus > 1:
        #python -m torch.distributed.launch --nproc_per_node=4 --use_env train_classification_imdb.py run --backend=gloo
        #additional_command = 'run --backend=gloo'
        if select == 1:
            command += f" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0"
            command += f" --nnodes=1 --nproc_per_node={ncpus} --max-restarts=3"
        else:
            command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"
            command += f" --max-restarts=3 --rdzv-id=$JOB_ID"
            command += f" --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"
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
    select = kwargs.get('select', 1)  # number of nodes
    ngpus, ncpus = int(ngpus), int(ncpus)
    singularity_path = "../built_containers/FaContainer_v2.sif"
    command, additional_command, train_with_ddp = command_setup(ngpus, ncpus, singularity_path,
                                                                select=select)    

    from qsub_parser import qsub, job_divider
    from path_setup import project_ls
    #project_ls = ["ddl"]  # can add more projects here
    pbs_array_data = get_pbs_array_data(kwargss)    
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        kwargs_qsub = {"path":kwargs.get("job_path"),  # acts as PBSout
                       "P":project_ls[pidx],
                       "ngpus":ngpus, 
                       "ncpus":ncpus, 
                       "select":select,
                       #"walltime":'95:59:59', 
                       #"walltime":'71:59:59',
                       #"walltime":'59:59:59',
                       "walltime":'35:59:59',
                       #"walltime":'39:59:59',
                       #"walltime":'35:59:59',
                       #"walltime":'29:59:59',
                       #"walltime":'23:59:59',
                       #"walltime":'23:59:59',
                       "mem":"48GB"
                       #"mem":"16GB"
                       } 
        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        print("\n")

if __name__ == '__main__':

    # script for running
    script_name = "main_seq_classification.py" 
    #dataset_names = ['imdb']  # add or change datasets here
    dataset_names = ['rotten_tomatoes']
    
    debug_mode = False
    for dataset_name in dataset_names:
        if not debug_mode:
            ngpus, ncpus = 0, 12
            select = 1
            train_with_ddp = True if max(ngpus, ncpus) > 1 else False   

            # empty dict is diffuser
            #kwargss = [{"with_frac":True, "gamma":0.2}, {"with_frac":True, "gamma":0.4},
            #           {"with_frac":True, "gamma":0.6}, {"with_frac":True, "gamma":0.8}]  
            #kwargss = [{"with_frac":True, "gamma":0.25}, 
            #           {"with_frac":True, "gamma":0.5}, {"with_frac":True, "gamma":0.75}]  
            kwargss = [{"with_frac":True, "gamma":0.5, "lr":3e-4}, {"with_frac":True, "gamma":0.5, "lr":3e-3}]    
            #kwargss = [{"with_frac":True, "gamma":0.2}, {"with_frac":True, "gamma":0.8}]                              
            #kwargss = [{}, {"with_frac":True, "gamma":0.4}]                         
            #model_root_dir = join(droot, "renorm_tomatoes")  # W := exp(Q^T K)
            #model_root_dir = join(droot, "invrenorm_tomatoes")  # W := exp(-Q^T K)
            #model_root_dir = join(droot, "reg_tomatoes")  # regularized embedding
            #model_root_dir = join(droot, "dereg_tomatoes")  # regularized embedding (detached frac lapl)
            #model_root_dir = join(droot, "derhoandoutdegreg_tomatoes")  # regularized embedding (detached rho and out-degree)
            model_root_dir = join(droot, "derhoreg_lrtest_tomatoes")  # regularized embedding (detached rho, droot/derhoreg_lrtest_tomatoes)
            common_kwargs = {"gradient_accumulation_steps":4,
                             "epochs":20,
                             "warmup_steps":50,
                             "divider": 1,
                             "per_device_train_batch_size":4,
                             "per_device_eval_batch_size":4}
        else:
            ngpus, ncpus = 0, 2
            select = 1
            train_with_ddp = True if max(ngpus, ncpus) > 1 else False   

            kwargss = [{}, {"with_frac":True, "gamma":0.8, "lr":3e-4}]  
            model_root_dir = join(droot, "debug_mode14")
            common_kwargs = {"gradient_accumulation_steps":2,
                             "divider": 100,
                             "max_steps": 200,
                             "warmup_steps":0, "eval_steps":50, "logging_steps":50, "save_steps":50,
                             "per_device_train_batch_size":2,
                             "per_device_eval_batch_size":2} 
        if train_with_ddp:
            common_kwargs["train_with_ddp"] = train_with_ddp
        kwargss = add_common_kwargs(kwargss, common_kwargs)
        
        for idx in range(len(kwargss)):
            # function automatically creates dir
            kwargss[idx]["dataset_name"] = dataset_name
            models_dir, kwargss[idx]["model_dir"] = create_model_dir(model_root_dir, **kwargss[idx])         

        print(kwargss)        
        train_submit(script_name, ngpus, ncpus, kwargss,
                     select=select, 
                     job_path=model_root_dir)