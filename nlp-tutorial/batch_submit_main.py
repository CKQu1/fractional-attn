import argparse
from datetime import datetime
from constants import DROOT, CLUSTER, MODEL_SUFFIX
from UTILS.mutils import njoin, get_seed, structural_model_root, str2bool
from qsub_parser import job_setup, qsub, add_common_kwargs

from batch_exps import *  # exp configs

"""
torchrun --nnodes=1 --nproc_per_node=2 ddp_main.py --max_iters=5 --eval_interval=5\
 --eval_iters=200 --weight_decay=0 --n_layers=1 --n_attn_heads=2
"""

if __name__ == '__main__':
          
    parser = argparse.ArgumentParser(description='batch_submit_main.py args')   
    parser.add_argument('--is_qsub', type=str2bool, nargs='?', const=True, default=False) 
    args = parser.parse_args()

    date_str = datetime.today().strftime('%Y-%m-%d')    
    batch_script_name = "batch_main.py"

    exp_type = 'exp2'
    if exp_type == 'exp1':                           # train full-sized models
        EXPS_TO_RUN = train_exps_full()
    if exp_type == 'exp2':                           # train models of depth 1, 2 and 3
        EXPS_TO_RUN = train_exps_hyperparam()        
    elif exp_type == 'exp3':                         # dynamic inference
        EXPS_TO_RUN = dynamic_inference_exps()
    elif exp_type == 'exp4':                         # attn graph from pretrained models
        EXPS_TO_RUN = attn_graph_exps()
    
    kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack = EXPS_TO_RUN

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
    
    if args.is_qsub:
        print(f'----- SUBMITTING ----- \n')
        for i in range(len(commands)):
            qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])         