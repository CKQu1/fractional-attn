import os
from os.path import isfile, join
from path_setup import droot

# main dir
repo_dir = os.getcwd()

def command_setup(ngpus, ncpus, singularity_path):
    assert isfile(singularity_path), "singularity_path does not exist!"

    if len(singularity_path) > 0:
        command = f"singularity exec --home {repo_dir} {singularity_path}"
    else:
        command = ""

    additional_command = ''
    train_with_ddp = False
    if ngpus <= 1 and ncpus <= 1:
        command += f" python"
    elif ngpus > 1:
        command += f" CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node={ngpus}"
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

def train_submit(script_name, ngpus, ncpus, kwargss):
    assert isfile(script_name), f"{script_name} does not exist!"

    # computing resource settings
    ngpus, ncpus = int(ngpus), int(ncpus)
    singularity_path = "../built_containers/FaContainer_v2.sif"
    command, additional_command, train_with_ddp = command_setup(ngpus, ncpus, singularity_path)    

    from qsub_parser import qsub, job_divider
    project_ls = ["ddl"]  # can add more projects here
    pbs_array_data = get_pbs_array_data(kwargss)
    #print(pbs_array_data)
    #print(len(pbs_array_data))        
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        kwargs_qsub = {#"path":join(droot, "trained_seq_classification"), 
                       #"path":join(droot, "qsub_parser_test"),
                       "path":join(droot, "main_seq_classification"),
                       "P":project_ls[pidx],
                       "ngpus":ngpus, 
                       "ncpus":ncpus, 
                       "walltime":'23:59:59', 
                       "mem":"20GB"}        
        if len(additional_command) > 0:
            kwargs_qsub["additional_command"] = additional_command

        qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   

if __name__ == '__main__':

    # script for running
    #script_name = join(repo_dir, "main_seq_classification.py")
    script_name = "main_seq_classification.py"
    ngpus, ncpus = 0, 4
    train_with_ddp = True
    kwargss = [ {"with_frac":False}, {"with_frac":True, "gamma":0.5} ]
    # "qsub_parser_test"
    common_kwargs = {"gradient_accumulation_steps":1, "model_dir":join(droot, "main_seq_classification"),
                     "epochs": 0.1,
                     "warmup_steps":10, "eval_steps":50, "logging_steps":50, "save_steps":50,
                     "per_device_eval_batch_size":2,
                     "train_with_ddp":train_with_ddp}
    kwargss = add_common_kwargs(kwargss, common_kwargs)

    train_submit(script_name, ngpus, ncpus, kwargss)