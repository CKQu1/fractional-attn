import numpy as np
import sys
import os
import random
from os.path import isdir, isfile
from constants import *
from mutils import njoin

def qsub(command, pbs_array_data, **kwargs):
    global pbs_array_data_chunks

    system = kwargs.get('system')
    nstack = kwargs.get('nstack', 1)

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
    
    # ---------- begin{ARTEMIS} ----------
    if system == 'ARTEMIS':
        MAX_SUBJOBS = 1000
        # PBS array jobs are limited to 1000 subjobs by default
        pbs_array_data_chunks = [pbs_array_data[x:x+MAX_SUBJOBS]
                                for x in range(0, len(pbs_array_data), MAX_SUBJOBS)]
        if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
            pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
        for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):

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
                #PBS -J {MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_chunk)-1}
                args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{MAX_SUBJOBS*i}])))" $PBS_ARRAY_INDEX))
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

            os.system(f'qsub {PBS_SCRIPT}')
            #print(PBS_SCRIPT)

    # ---------- end{ARTEMIS} ----------s

    # ---------- begin{PHYSICS1} ----------
    elif system == 'PHYSICS':
        MAX_SUBJOBS = 1000
        
        for i, pbs_array_data_point in enumerate(pbs_array_data):

            # old args
            # args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_point})))"))
            # echo "pbs_array_args = ${{args[*]}}"
            # {command} ${{args}} {additional_command} {post_command}

            args = ' '.join(map(str, pbs_array_data_point))
            full_command = f'{command} {args} {additional_command} {post_command}'

            # https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
            PBS_SCRIPT = f"""<<eof
                #!/bin/bash
                #PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
                #PBS -q {kwargs.get('q','defaultQ')}
                #PBS -V
                #PBS -m n
                #PBS -o {path} -e {path}
                #PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
                #PBS -l walltime={kwargs.get('walltime','23:59:00')}                       
                
                #cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
                echo "pbs_array_args = ${args}"
                # if [ {source_exists} ]; then
                #     source {kwargs.get('source')}
                # fi
                # if [ {conda_exists} ]; then
                #     conda activate {kwargs.get('conda')}
                # fi         

                cd fractional-attn/long-range-arena
                source /usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh
                conda activate frac_attn                                                        
                {full_command}
                exit
            eof"""

            os.system(f'qsub {PBS_SCRIPT}')
            #print(PBS_SCRIPT)

    # ---------- end{PHYSICS1} ----------

    # ---------- begin{PHYSICS2} ----------
    # elif system == 'PHYSICS':
    #     MAX_SUBJOBS = 1000
        
    #     # ##PBS -J {MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_point)-1}
    #     PBS_SCRIPT_TEMPLATE = f"""<<eof
    #         #!/bin/bash
    #         #PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
    #         #PBS -q {kwargs.get('q','defaultQ')}
    #         #PBS -V
    #         #PBS -m n
    #         #PBS -o {path} -e {path}
    #         #PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
    #         #PBS -l walltime={kwargs.get('walltime','23:59:00')}                                   
            

    #         #cd {kwargs.get('cd', '$PBS_O_WORKDIR')}            
    #         # if [ {source_exists} ]; then source {kwargs.get('source')} ; fi;
    #         # if [ {conda_exists} ]; then conda activate {kwargs.get('conda')} ; fi;    
    #         cd fractional-attn/long-range-arena
    #         source /usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh
    #         conda activate frac_attn                                                        
    #         """
    #     PBS_SCRIPT_END = '\n echo "pbs_array_args = ${args}" \n exit \n eof'

    #     PBS_SCRIPT = ''
    #     for i, pbs_array_data_point in enumerate(pbs_array_data):

    #         # old args
    #         # args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_point})))"))
    #         # echo "pbs_array_args = ${{args[*]}}"
    #         # {command} ${{args}} {additional_command} {post_command}

    #         args = ' '.join(map(str, pbs_array_data_point))

    #         # https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash            

    #         full_command = f'\n {command} {args} {additional_command} {post_command}'
    #         if len(PBS_SCRIPT) == 0:
    #             PBS_SCRIPT = PBS_SCRIPT_TEMPLATE + full_command
    #         else:
    #             PBS_SCRIPT += full_command

    #         if i == len(pbs_array_data) - 1 or (i % nstack == nstack - 1):                                
    #             PBS_SCRIPT += PBS_SCRIPT_END

    #             os.system(f'qsub {PBS_SCRIPT}')
    #             #print(PBS_SCRIPT)

    #             PBS_SCRIPT = ''            
                                
    # ---------- end{PHYSICS2} ----------

    # ---------- begin{PHYSICS3} ----------
    # elif system == 'PHYSICS':
    #     MAX_SUBJOBS = 1000
    #     # PBS array jobs are limited to 1000 subjobs by default
    #     pbs_array_data_chunks = [pbs_array_data[x:x+MAX_SUBJOBS]
    #                              for x in range(0, len(pbs_array_data), MAX_SUBJOBS)]
    #     if len(pbs_array_data_chunks[-1]) == 1:  # array jobs must have length >1
    #         pbs_array_data_chunks[-1].insert(0, pbs_array_data_chunks[-2].pop())
    #     for i, pbs_array_data_chunk in enumerate(pbs_array_data_chunks):
        
    #         # old args          
    #         # args=($(python -c "import sys;print(' '.join(map(str, {pbs_array_data_point})))"))  
    #         # echo "pbs_array_args = ${{args[*]}}"
    #         # {command} ${{args}} {additional_command} {post_command}

    #         # https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
    #         PBS_SCRIPT = f"""<<eof
    #             #!/bin/bash
    #             #PBS -N {kwargs.get('N', sys.argv[0] or 'job')}
    #             #PBS -q {kwargs.get('q','defaultQ')}
    #             #PBS -V
    #             #PBS -m n
    #             #PBS -o {path} -e {path}
    #             #PBS -l select={kwargs.get('select',1)}:ncpus={kwargs.get('ncpus',1)}:mem={kwargs.get('mem','1GB')}{':ngpus='+str(kwargs['ngpus']) if 'ngpus' in kwargs else ''}
    #             #PBS -l walltime={kwargs.get('walltime','23:59:00')}                       
    #             #PBS -J {MAX_SUBJOBS*i}-{MAX_SUBJOBS*i + len(pbs_array_data_chunk)-1}                                

    #             args=($(python -c $PBS_ARRAY_INDEX "import sys;print(' '.join(map(str, {pbs_array_data_chunk}[int(sys.argv[1])-{MAX_SUBJOBS*i}])))"))

    #             cd {kwargs.get('cd', '$PBS_O_WORKDIR')}
    #             echo "pbs_array_args = ${{args[*]}}"     

    #             # cd fractional-attn/long-range-arena
    #             source /usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh
    #             conda activate frac_attn                                                        
    #             {command} ${{args[*]}} {additional_command} {post_command}
    #             exit
    #         eof"""

    #     os.system(f'qsub {PBS_SCRIPT}')
    #     #print(PBS_SCRIPT)

    # ---------- end{PHYSICS3} ----------



def job_setup(script_name, kwargss, **kwargs):

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
    commands, script_names, pbs_array_trues, kwargs_qsubs = [], [], [], []
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
        #print("\n")
        commands.append(command)
        script_names.append(script_name)
        pbs_array_trues.append(pbs_array_true)
        kwargs_qsubs.append(kwargs_qsub)

    assert len(list(set([len(commands), len(script_names), len(pbs_array_trues), len(kwargs_qsubs)]))) == 1, 'len not same'
    return commands, script_names, pbs_array_trues, kwargs_qsubs

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

def command_setup_ddp(**kwargs):

    system = kwargs.get('system')
    ncpus = kwargs.get('ncpus', 1) 
    ngpus = kwargs.get('ngpus', 0)
    select = kwargs.get('select', 1)
    
    if 'singularity_path' in kwargs:
        singularity_path = kwargs.get('singularity_path')
        assert isfile(singularity_path), "singularity_path does not exist!"
    else:
        singularity_path = ''

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

# originall in submit_main.py

"""
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

        qsub(f'{command} {script_name}', pbs_array_true, **kwargs_qsub)   
        print("\n")
"""  