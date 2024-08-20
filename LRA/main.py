import os
import sys
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import redirect_stdout
#from config import Config
from config_v2 import Config
from models.model_LRA import ModelForSC, ModelForSCDual
from models.dataset_LRA import LRADataset

from os.path import isdir, isfile
from os import makedirs
from constants import *
from mutils import *

# ----- device -----
dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                    if torch.cuda.is_available() else "cpu") 
# ------------------

def print_summary(summary, save_if_improved, model, checkpoint_path):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])


    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]

    summary_round = {}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

def step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
             accumu_steps, init_t, summary, component, step_idx, writer=None):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].to(dev)

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            # with torch.cuda.amp.autocast():
            partial_outputs = model(**partial_inputs)

            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]


            amp_scaler.scale(partial_outputs["loss"]).backward() # loss.backward()

        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        amp_scaler.unscale_(optimizer)



        nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping


        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    if step_idx%100==0:
        print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

    if writer is not None:
        writer.add_scalar('loss', loss, step_idx)
        writer.add_scalar('accu', accu, step_idx)

    return outputs

def train_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
              training_config, summary, writer):

    accumu_steps = training_config['accumu_steps']
    checkpoint_path = training_config['checkpoint_path']
    # best_dev_loss = float(1e10)
    best_dev_accu = 0
    total_step = training_config["num_train_steps"]

    init_t = time.time()

    model.train()
    for train_step_idx in range(total_step):
        outputs = step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
                           accumu_steps, init_t, summary, component='train', step_idx=train_step_idx,writer=writer)

        if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
            print_summary(summary["train"], False, model, checkpoint_path)
            model.eval()
            for dev_step_idx in range(training_config["num_eval_steps"]):
                outputs = step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
                                   accumu_steps, init_t, summary, component='dev', step_idx=dev_step_idx)
            # dev_loss = np.mean(summary["dev"]["loss"])
            # if  dev_loss < best_dev_loss:
            #     best_dev_loss = dev_loss
            dev_accu = np.mean(summary["dev"]["accu"])
            if dev_accu > best_dev_accu:
                best_dev_accu = dev_accu
                if (train_step_idx + 1) > total_step * 0.2:
                    torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
                    print('best model saved: step = ',train_step_idx, 'dev accu = ',dev_accu)

            print_summary(summary["dev"], True, model, checkpoint_path)
            model.train()





    print('total training step (k): {}'.format(total_step/1000.0))
    print("total training time (s): {}".format(int(time.time()-init_t)))
    if "cpu" not in dev.type:
        print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))


def eval_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
             training_config, summary):
    accumu_steps = training_config['accumu_steps']
    checkpoint_path = training_config['checkpoint_path']
    init_t = time.time()
    model.eval()
    try:
        for test_step_idx in itertools.count():
            outputs = step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
                               accumu_steps, init_t, summary, component='test', step_idx=test_step_idx)
    except StopIteration:
        print_summary(summary["test"], False, model, checkpoint_path)

if __name__ == '__main__':
    #main()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default="train",
                        help="train eval")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument("--attn", type = str, default="softmax",
                        help = "softmax, opfns, sink")
    parser.add_argument("--task", type = str, default="lra-listops",
                        help = "lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=42)

    parser.add_argument('--log_dir', type=str, default='')

    # ----- ADDED -----
    # FNS
    parser.add_argument("--manifold", type=str, default='sphere')
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0)
    # SINK
    parser.add_argument("--n_it", type=int, default=1)
    # General
    parser.add_argument('--qkv_bias', type=str2bool, nargs='?', const=True, default=False)
    # Training
    parser.add_argument("--lr_scheduler", type=str, default='onecyclelr', help="onecyclelr | constantlr")
    # -----------------

    args = parser.parse_args()    

    # -------------------------------------------------------------------------------------------------------------

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'

    assert args.task in TASKS, f'{args.task} does not exist!'
    assert args.attn in ['opfns', 'sink', 'softmax'], f'{args.attn} attn doese not exist!'

    ### get model config ###
    model_config = Config[args.task]["model"]
    if args.attn in Config[args.task]["extra_attn_config"]:
        model_config.update(Config[args.task]["extra_attn_config"][args.attn])
    model_config["mixed_precision"] = True
    model_config["attn_type"] = args.attn
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random    
    # ----- ADDED -----
    if args.attn in ['fns', 'opfns']:
        assert args.manifold in ['sphere', 'rd'], 'FNS manifold: sphere or rd'   
        assert 1 <= args.alpha <= 2, 'FNS alpha must be between [1,2]'
        assert args.a in [0,0.5,1], 'Normalization index must be 0 or 0.5 or 1'  

        model_config["manifold"] = args.manifold
        model_config["alpha"] = args.alpha
        model_config["bandwidth"] = args.bandwidth
        model_config["a"] = args.a

        if args.manifold == 'sphere':

            if args.alpha < 2:
                if Config[args.task]['model']['num_head'] == 1:
                    model_config['d_intrinsic'] = Config[args.task]['model']['head_dim'] - 1
                else:
                    model_config['d_intrinsic'] = Config[args.task]['model']['head_dim']
                model_config['sphere_radius'] = ((np.pi**(1/model_config['d_intrinsic'])-1)/np.pi)                
            elif args.alpha >= 2:
                model_config['sphere_radius'] = 1                 
        
            # mask for distance
            model_config['mask_val'] = model_config['sphere_radius'] * np.pi

            model_config["attn_type"] = 'sp' + args.attn

        elif args.manifold == 'rd':
            if args.alpha < 2:
                model_config['d_intrinsic'] = Config[args.task]['model']['head_dim']  # head_dim                
            #model_config['mask_val'] = 1 / model_config["max_length"]
            model_config['mask_val'] = 1e-5

            model_config["attn_type"] = 'rd' + args.attn

        # degree index
        model_config['a'] = args.a  

    elif args.attn == 'sink':
        model_config['n_it'] = args.n_it                   
        model_config["bandwidth"] = args.bandwidth
        model_config['mask_val'] = -1e9

    # general
    model_config['qkv_bias'] = args.qkv_bias

    # -----------------

    training_config = Config[args.task]["training"]    
    ### log preparation ###
    # log_dir = './log-{}/'.format(args.random)  
    if args.log_dir == '':        
        _, log_dir = create_model_dir(njoin(DROOT, 'trained_models'), attn=args.attn, 
                                      task=args.task.split('-')[1], **model_config)
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print(f'log_dir = {log_dir}')

    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.checkpoint))
    redirect_stdout(open(log_path, 'w'))
    summary = {
        component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
        for component in ["train", "dev", "test"]
    }
    writer = SummaryWriter(os.path.join(log_dir,'{}.tensorboard'.format(args.checkpoint)))

    print(f"Task = {args.task}")
    print(json.dumps([model_config, training_config], indent = 4))


    ###  set the random seeds for deterministic results. ####
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True



    ### model preparation ###
    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)


    #checkpoint_dir = './checkpoints-{}'.format(args.random)
    checkpoint_dir = njoin(log_dir, './checkpoints-{}'.format(args.random))
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = njoin(checkpoint_dir, '{}.{}.model'.format(args.checkpoint, args.random))
    training_config["checkpoint_path"] = checkpoint_path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("model loaded from: " + checkpoint_path)

    model = model.to(dev)
    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    ### Use DP ###
    if "cpu" not in dev.type:
        device_ids = list(range(torch.cuda.device_count()))
        #device_ids = int(os.environ["WORLD_SIZE"])
        print(f"GPU list: {device_ids}")
        #if len(device_ids) > 1:    
        model = nn.DataParallel(model, device_ids = device_ids)

    ### data preparation ###

    # if isdir('data'):
    #     data_root = 'data'
    # else:
    #     from pathlib import Path
    #     data_root = njoin('./'.parent.absolute(), 'data')
    data_root = DROOT
    ds_iter = {
        "train":enumerate(DataLoader(LRADataset(njoin(data_root,"lra_processed", f"{args.task}.train.pickle"), True), 
        batch_size = training_config["batch_size"], drop_last = True)),
        "dev":enumerate(DataLoader(LRADataset(njoin(data_root,"lra_processed", f"{args.task}.dev.pickle"), True), 
        batch_size = training_config["batch_size"], drop_last = True)),
        "test":enumerate(DataLoader(LRADataset(njoin(data_root,"lra_processed", f"{args.task}.test.pickle"), False), 
        batch_size = training_config["batch_size"], drop_last = True)),
    }        

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
    )

    if args.lr_scheduler == 'onecyclelr':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer,
            max_lr = training_config["learning_rate"],
            pct_start = training_config["warmup"] / training_config["num_train_steps"],
            anneal_strategy = training_config["lr_decay"],
            total_steps = training_config["num_train_steps"]
        )

    elif args.lr_scheduler == 'constantlr':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer = optimizer,        
            total_iters = int(training_config["num_train_steps"] / 3)
        )        

    amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None


    # accumu_steps = max(training_config["batch_size"] // len(device_ids) // model_config["gpu_memory"], 1)
    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    # accumu_steps = 1
    print(f"accumu_steps={accumu_steps}")
    training_config['accumu_steps'] = accumu_steps



    ### train ###
    if args.mode == 'train':
        train_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
                  training_config, summary, writer)

    ### eval ###
    if os.path.exists(checkpoint_path) and checkpoint_path != './checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
    eval_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
             training_config, summary)    
