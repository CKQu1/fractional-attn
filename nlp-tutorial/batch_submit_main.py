import argparse
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
    parser.add_argument('--exp', default='exp1', type=str) 
    parser.add_argument('--nstack', default=1, type=int)
    args = parser.parse_args()
    
    batch_script_name = "batch_main.py"
    nstack = args.nstack

    exp_type = args.exp
    if exp_type == 'exp1':                           # train full-sized models (R^d)
        EXPS_TO_RUN = train_exps_full(nstack, manifold='rd'); EXP_NAME = '6-layer model training for Euclidean case'
    elif exp_type == 'exp2':                           # train models of depth 1, 2 and 3
        EXPS_TO_RUN = train_exps_hyperparam(nstack); EXP_NAME = 'hyperparam model training'        
    elif exp_type == 'exp3':                         # dynamic inference for small models
        EXPS_TO_RUN = dynamic_inference_small(nstack, is_dist_based=False)
        EXP_NAME = 'dynamic inference (small models)'
    elif exp_type == 'exp4':                         # dynamic inference for large models
        EXPS_TO_RUN = dynamic_inference_small(nstack, is_dist_based=True)
        EXP_NAME = 'locality analysis (small models)'
    elif exp_type == 'exp5':                         # spectral gap from pretrained models
        EXPS_TO_RUN = attn_graph_exps(nstack, script_name='attn_graph_v2.py'); EXP_NAME = 'spectral gap'
    elif exp_type == 'exp6':                         # attn graph from pretrained models
        EXPS_TO_RUN = attn_graph_exps(nstack, script_name='attn_graph_final.py'); EXP_NAME = 'attn graph reconstruction'
    elif exp_type == 'exp7':                           # train full-sized models (sphere)
        EXPS_TO_RUN = train_exps_full(nstack, manifold='sphere'); EXP_NAME = '6-layer model training for spherical case'
    elif exp_type == 'exp8':
        EXPS_TO_RUN = train_exps_hyperdataset(nstack); EXP_NAME = 'hyperdataset model training'

    print('-----------------------')
    print(f'{exp_type}: {EXP_NAME}')
    print('----------------------- \n')
    
    kwargss_all, script_name, q, ncpus, ngpus, select, walltime, mem, job_path, nstack = EXPS_TO_RUN

    # ----- submit jobs -----
    print(f'Total jobs: {len(kwargss_all)} \n')      

    def expand_resource(resource, resource_name):
        if isinstance(resource, (list, tuple)):
            assert len(resource) == len(kwargss_all), f'{resource_name} must match kwargss_all'
            return list(resource)
        return [resource] * len(kwargss_all)

    def make_batched_kwargs(kwargss):
        batch_kwargss = []
        kwargsss = [kwargss[i:i+nstack] for i in range(0, len(kwargss), nstack)]
        for kwargss_batch in kwargsss:
            arg_strss = ''
            for kwargs in kwargss_batch:
                arg_strss += ",".join("=".join((str(k),str(v))) for k,v in kwargs.items()) + ';'
            batch_kwargss.append({'arg_strss': arg_strss[:-1], 'script': script_name})
        return batch_kwargss

    walltimes = expand_resource(walltime, 'walltime')
    mems = expand_resource(mem, 'mem')

    resource_groups = {}
    for kwargs, job_walltime, job_mem in zip(kwargss_all, walltimes, mems):
        resource_groups.setdefault((job_walltime, job_mem), []).append(kwargs)

    resource_batches = []
    for (job_walltime, job_mem), kwargss in resource_groups.items():
        resource_batches.append((job_walltime, job_mem, make_batched_kwargs(kwargss)))

    print(f'Batched Total jobs: {sum(len(batch_kwargss) for _, _, batch_kwargss in resource_batches)} \n')

    if len(resource_batches) > 1:
        print('Resource groups:')
        for job_walltime, job_mem, batch_kwargss in resource_batches:
            print(f'  walltime={job_walltime}, mem={job_mem}: {len(batch_kwargss)} batched jobs')
        print()

    commands, batch_script_names, pbs_array_trues, kwargs_qsubs = [], [], [], []
    for job_walltime, job_mem, batch_kwargss_all in resource_batches:
        group_commands, group_batch_script_names, group_pbs_array_trues, group_kwargs_qsubs =\
                job_setup(batch_script_name, batch_kwargss_all,
                        q=q,
                        ncpus=ncpus,
                        ngpus=ngpus,
                        select=select,
                        walltime=job_walltime,
                        mem=job_mem,
                        job_path=job_path,
                        nstack=nstack,
                        cluster=CLUSTER)

        commands += group_commands
        batch_script_names += group_batch_script_names
        pbs_array_trues += group_pbs_array_trues
        kwargs_qsubs += group_kwargs_qsubs
    
    if args.is_qsub:
        print(f'----- SUBMITTING ----- \n')
        for i in range(len(commands)):
            qsub(f'{commands[i]} {batch_script_names[i]}', pbs_array_trues[i], path=job_path, **kwargs_qsubs[i])         
