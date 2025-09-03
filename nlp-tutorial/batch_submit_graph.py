import os
from datetime import datetime
from itertools import product
from qsub_parser import job_setup, qsub, add_common_kwargs

import re
from pathlib import Path

if __name__ == '__main__':

    date_str = datetime.today().strftime('%Y-%m-%d')    
    batch_script_name = "batch_main.py"

    # exp configs
    kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack = attn_graph_exps()

    # ----- submit jobs -----
    print(f'Total jobs: {len(kwargss_all)} \n')      

    batch_kwargss_all = []
    kwargsss = [kwargss_all[i:i+nstack] for i in range(0, len(kwargss_all), nstack)]
    for kwargss in kwargsss:
        arg_strss = ''
        for kwargs in kwargss:
            arg_strss += ",".join("=".join((str(k),str(v))) for k,v in kwargs.items()) + ';'
        batch_kwargss_all.append({'arg_strss': arg_strss[:-1], 'script': script_name})

    print(f'Batched Total jobs: {len(batch_kwargss_all)} \n')

    commands, batch_script_names, pbs_array_trues, kwargs_qsubs =\
            job_setup(batch_script_name, batch_kwargss_all,
                    q=q,
                    ncpus=ncpus,
                    ngpus=ngpus,
                    select=select, 
                    walltime=walltime,
                    mem=mem,                    
                    job_path=job_path,
                    nstack=nstack,
                    cluster=CLUSTER)
    
    for i in range(len(commands)):
        qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])      