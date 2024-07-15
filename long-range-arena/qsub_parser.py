import numpy as np
import sys
import os
import random
from os.path import isdir, isfile
from constants import BPATH
from mutils import njoin

def qsub(command, pbs_array_data, **kwargs):

    system = kwargs.get('system')

    if 'path' in kwargs:
        path = kwargs['path']
        if path and path[-1] != os.sep: path += os.sep
    else:
        path = command.replace(' ', '_') + os.sep
    if 'additional_command' in kwargs:
        additional_command = kwargs['additional_command']
    else:
        additional_command = ''
    if kwargs.get('pass_path', False):
        post_command = path if kwargs['pass_path'] == True else kwargs['pass_path']
    else:
        post_command = ''
    # Create output folder.
    if not isdir(njoin(path,"job")): os.makedirs(njoin(path,"job"))
    # source virtualenv
    if 'source' in kwargs:
        assert os.path.isfile(kwargs.get('source')), "source for virtualenv incorrect"
        source_exists = 'true'
    else:
        source_exists = 'false'
    # conda activate
    if 'conda' in kwargs:
        conda_exists = 'true'
    else:
        conda_exists = 'false'
    if kwargs.get('local', False):  # Run the subjobs in the current process.
        for pbs_array_args in pbs_array_data:
            str_pbs_array_args = ' '.njoin(map(str, pbs_array_args))
            os.system(f"""bash <<'END'
                cd {kwargs.get('cd', '.')}
                echo "pbs_array_args = {str_pbs_array_args}"
                {command} {str_pbs_array_args} {post_command}
END""")
        return
    # Distribute subjobs evenly across array chunks.
    pbs_array_data = random.sample(pbs_array_data, len(pbs_array_data))
    # Submit array job.
    print(f"Submitting {len(pbs_array_data)} subjobs")
    # PBS array jobs are limited to 1000 subjobs by default
    pbs_array_data_chunks = [pbs_array_data[x:x+1000]
                             for x in range(0, len(pbs_array_data), 1000)]
    if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
        pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
    for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):

        # ---------- begin{ARTEMIS} ----------
        if system == 'ARTEMIS':
            PBS_SCRIPT = f"""<<'END'
                #!/bin/bash
                #PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
                #PBS -P {kwargs.get('P',"''")}
                #PBS -q {kwargs.get('q','defaultQ')}
                #PBS -V
                #PBS -m n
                #PBS -o {path}job -e {path}job
                #PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
                #PBS -l walltime={kwargs.get('walltime','23:59:00')}
                #PBS -J {1000*i}-{1000*i + len(pbs_array_data_chunk)-1}
                args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{1000*i}])))" $PBS_ARRAY_INDEX))
                cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
                echo "pbs_array_args = ${{args[*]}}"
                #if [ {source_exists} ]; then
                #    source {kwargs.get('source')}
                #fi
                #if [ {conda_exists} ]; then
                #    conda activate {kwargs.get('conda')}
                #fi
                {command} ${{args[*]}} {additional_command} {post_command}
    END"""
        # ---------- end{ARTEMIS} ----------

        # ---------- begin{PHYSICS} ----------
        if system == 'PHYSICS':
            PBS_SCRIPT = f"""<<'END'
                #!/bin/bash
                #PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
                #PBS -q {kwargs.get('q','defaultQ')}
                #PBS -V
                #PBS -m n
                #PBS -o {path}job -e {path}job
                #PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
                #PBS -l walltime={kwargs.get('walltime','23:59:00')}
                #PBS -J {1000*i}-{1000*i + len(pbs_array_data_chunk)-1}
                args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{1000*i}])))" $PBS_ARRAY_INDEX))
                cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
                echo "pbs_array_args = ${{args[*]}}"
                #if [ {source_exists} ]; then
                #    source {kwargs.get('source')}
                #fi
                #if [ {conda_exists} ]; then
                #    conda activate {kwargs.get('conda')}
                #fi
                {command} ${{args[*]}} {additional_command} {post_command}
    END"""
        # ---------- end{PHYSICS} ----------

        os.system(f'qsub {PBS_SCRIPT}')
        #print(PBS_SCRIPT)

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

# N is the total number of projects
def job_divider(pbs_array: list, N: int):
    total_jobs = len(pbs_array)
    assert total_jobs >= 2, 'total_jobs does not exceed 1'
    ncores = min(int(np.floor(total_jobs/2)), N)
    pbss = []
    delta = int(round(total_jobs/ncores))
    for idx in range(ncores):
        if idx != ncores - 1:
            pbss.append( pbs_array[idx*delta:(idx+1)*delta] )
        else:
            if len(pbs_array[idx*delta:]) < 2:
                pbss[-1] = pbss[-1] + pbs_array[idx*delta:]
            else:    
                pbss.append( pbs_array[idx*delta:] )   
    ncores = len(pbss)
    perm = list(np.random.choice(N,ncores,replace=False))
    assert len(perm) == len(pbss), "perm length and pbss length not equal!"

    return perm, pbss

def command_setup(singularity_path, **kwargs):
    assert isfile(singularity_path), "singularity_path does not exist!"

    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)
    bind_path = kwargs.get('bind_path', BPATH)
    home_path = kwargs.get('home_path', os.getcwd())
    if len(singularity_path) > 0:
        command = f"singularity exec --bind {bind_path} --home {home_path} {singularity_path}"
    else:
        command = ""

    additional_command = ''
    command += " python"

    if len(singularity_path) == 0:
        command = command[1:]

    return command, additional_command

def command_setup_ddp(singularity_path, **kwargs):
    assert isfile(singularity_path), "singularity_path does not exist!"

    system = kwargs.get('system')

    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)
    
    if len(singularity_path) > 0:
        bind_path = kwargs.get('bind_path', BPATH)
        home_path = kwargs.get('home_path', os.getcwd())        
        command = f"singularity exec --bind {bind_path} --home {home_path} {singularity_path}"
    else:
        command = ""

    additional_command = ''
    if max(ngpus, ncpus) <= 1:
        command += " python"
    elif ngpus > 1:
        #command += f" CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node={ngpus}"
        #additional_command = 'run --backend=nccl'
        if select == 1:
            # command += f" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0"
            # command += f" --nnodes=1 --nproc_per_node={ngpus} --max-restarts=3"
            command += f" torchrun --standalone --nnodes=1 --nproc_per_node={ncpus}"
        else:
            # tolerates 3 failures
            # command += f" torchrun --nnodes={select} --nproc_per_node={ngpus}"
            # command += f" --max-restarts=3 --rdzv-id=$JOB_ID"
            # command += f" --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"
            command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"
    elif ngpus == 0 and ncpus > 1:
        #python -m torch.distributed.launch --nproc_per_node=4 --use_env train_classification_imdb.py run --backend=gloo
        #additional_command = 'run --backend=gloo'
        if select == 1:
            # command += f" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0"            
            # command += f" --nnodes=1 --nproc_per_node={ncpus} --max-restarts=3"
            command += f" torchrun --standalone --nnodes=1 --nproc_per_node={ncpus}"
        else:
            # command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"
            # command += f" --max-restarts=3 --rdzv-id=$JOB_ID"
            # command += f" --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"    
            command += f" torchrun --nnodes={select} --nproc_per_node={ncpus}"          

    if len(singularity_path) == 0:
        command = command[1:]

    return command, additional_command    