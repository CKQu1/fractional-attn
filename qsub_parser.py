import numpy as np
import sys
import os
import random
from os.path import join

def qsub(command, pbs_array_data, **kwargs):

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
    if not os.path.isdir(join(path,"job")): os.makedirs(join(path,"job"))
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
            str_pbs_array_args = ' '.join(map(str, pbs_array_args))
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
        os.system(f'qsub {PBS_SCRIPT}')
        #print(PBS_SCRIPT)

# N is the total number of projects
def job_divider(pbs_array: list, N: int):
    total_jobs = len(pbs_array)
    ncores = min(int(np.floor(total_jobs/2)), N)
    pbss = []
    delta = int(round(total_jobs/ncores))
    for idx in range(ncores):
        if idx != ncores - 1:
            pbss.append( pbs_array[idx*delta:(idx+1)*delta] )
        else:
            pbss.append( pbs_array[idx*delta::] )    
    perm = list(np.random.choice(N,ncores,replace=False))
    assert len(perm) == len(pbss), "perm length and pbss length not equal!"

    return perm, pbss
