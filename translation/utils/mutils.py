import argparse
import json
import os
import pandas as pd
from os.path import join, normpath, isdir, isfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# -------------------- Path utils --------------------

def njoin(*args):
    return normpath(join(*args))

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2ls(s):
    if isinstance(s, list):
        return s
    elif isinstance(s, str):
        if ',' in s:
            return s.split(',')
        else: 
            return [s]

def find_subdirs(root_dir, matching_str):
    matches = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if matching_str in dirname.lower() and dirpath not in matches:
                matches.append(dirpath)
    return matches

def get_seed(dir, *args):  # for enumerating each instance of training
    #global start, end, instances, s_part

    if isdir(dir):
        seeds = []
        dirnames = next(os.walk(dir))[1]
        if len(dirnames) > 0:
            for dirname in dirnames:        
                #is_append = (len(os.listdir(njoin(dir, dirname))) > 0)  # make sure file is non-empty
                is_append = True
                for s in args:
                    is_append = is_append and (s in dirname)
                #print(f'{dirname} is_append: {is_append}')  # delete
                if is_append:  
                    #try:        
                    #for s_part in dirname.split(s):
                    assert "model=" in dirname, f'str model= not in {dirname}'
                    start = dirname.find("model=")
                    seeds.append(int(dirname[start+6:]))
                    #except:
                    #    pass       
            #print(seeds)  # delete
            return max(seeds) + 1 if len(seeds) > 0 else 0
        else:
            return 0
    else:
        return 0

# create model_dir which determines the seed for the specific model/training setting
def create_model_dir(model_root_dir, **kwargs):
    model_name = kwargs.get('model_name', 'fnsformer')    
    category = kwargs.get('category', 'default')
    is_op = kwargs.get('is_op')
    # if '_' in dataset_name:
    #     dataset_code = ''.join([s[0] for s in dataset_name.split('_')])
    # else:
    #     dataset_code = dataset_name
    # qk_share = kwargs.get('qk_share')
    # assert isinstance(qk_share,bool), f'qk_share is not bool, has value {qk_share}'

    dirname = '' if not is_op else 'op'
    #dirname = f'{model_name}-{dataset_code}'
    dirname += f'{model_name}-{category}'
    #dirname += '-qqv' if qk_share is True else '-qkv'  # qk weight-tying
    if model_name[-9:] == 'fnsformer':                 
        alpha = kwargs.get("alpha", 1)
        bandwidth = kwargs.get("bandwidth", 1)             
        dirname += f'-alpha={alpha}-eps={bandwidth}'
        if 'dm' in model_name:
            a = kwargs.get('a', 1)
            dirname += f'-a={a}'
        # if alpha < 2:
        #     d_intrinsic = kwargs.get('d_intrinsic')
        #     dirname += f'-dman={d_intrinsic}'
    elif model_name == 'sinkformer':
        bandwidth = kwargs.get("bandwidth", 1)     
        n_it = kwargs.get("n_it", 1)        
        dirname += f'-n_it={n_it}-eps={bandwidth}'

    models_dir = njoin(model_root_dir, dirname)
    # if 'instance' in kwargs:
    #     instance = kwargs.get('instance')
    # else:
    #     instance = get_instance(models_dir, 'model=')
    assert 'seed' in kwargs, 'seed not in kwargs'
    seed = kwargs.get('seed')
    model_dir = njoin(models_dir, f'model={seed}')        
       
    return models_dir, model_dir      

# creates structural model_root based on model/training setting
def structural_model_root(**kwargs):

    PATH_CONFIG_DICT = {'layers': 'n_layers', 'heads': 'n_attn_heads', 'hidden': 'hidden_size'}

    qk_share = kwargs.get('qk_share', False)
    affix = 'qqv' if qk_share==True else 'qkv'    

    # lr = kwargs.get('lr'); bs = kwargs.get('bs'); epochs = kwargs.get('epochs')
    
    model_root = ''
    for metric_name in PATH_CONFIG_DICT.keys():
        metric = PATH_CONFIG_DICT[metric_name]
        if metric in kwargs:
            metric_val = kwargs.get(metric)
            model_root += f'{metric_name}={metric_val}-'
    model_root += affix

    if 'max_iters' in kwargs:
        max_iters = kwargs.get('max_iters')
        model_root += f'-max_iters={max_iters}'
    #model_root = njoin(model_root, f'lr={lr}-bs={bs}-epochs={epochs}')            

    return model_root       

# collects pretrained model_dir
def collect_model_dirs(models_root, **kwargs):

    suffix = kwargs.get('suffix', 'former')  # i.e. 'former', 'vit'
    contained_strs = kwargs.get('contained_strs')

    model_dirs = []  # full path
    model_names = []  # i.e. spopfnsformer, dpformer, etc
    for dirname in os.listdir(models_root):
        if suffix in dirname:
            model_dirs.append(njoin(models_root, dirname))
            model_names.append(dirname.split('-')[0])

    model_names = list(set(model_names))  # make names unique

    DCT_ALL = {}
    for model_name in model_names:
        cur_model_dirs = [model_dir for model_dir in model_dirs if model_name in model_dir]
        if 'fns' in model_name.lower():
            cols = ['alpha', 'bandwidth', 'a']
        elif 'sink' in model_name.lower():
            cols = ['n_it', 'bandwidth']
        elif 'dp' in model_name.lower():
            cols = []    
        metrics = ['bleu', 'train_loss', 'val_loss']   
        metrics_dict = {}     
        for metric in metrics:
            metrics_dict[metric] = []        
        # cols_attn = ['qk_share', 'qkv_bias', 'dataset_name']
        cols_config = ['is_op', 'qkv_bias']
        # cols_train = ['steps_per_epoch']
        cols_other = ['ensembles', 'seeds', 'model_dir']
        #cols +=  cols_attn + metrics + cols_config + cols_train + cols_other
        cols +=  metrics + cols_config + cols_other

        df = pd.DataFrame(columns=cols)
        model_dir_dct = {}
        for model_dir in cur_model_dirs:
            ensembles = 0
            seeds = []            
            for seed_dir in os.listdir(model_dir):
                if 'model=' in seed_dir:                    
                    seed = int(seed_dir.split('=')[-1])
                    fpath = njoin(model_dir, seed_dir)
                    if isfile(njoin(fpath, 'scalars', 'lr.csv')):
                        ensembles += 1
                        seeds.append(seed)  
                        # bleu score
                        if isfile(njoin(fpath, 'scalars', 'bleu.csv')):
                            run = pd.read_csv(njoin(fpath, 'scalars', 'bleu.csv'), index_col=False)                         
                            metrics_dict['bleu'].append(run.iloc[run.index[-1],-1])                               
                        else:                            
                            metrics_dict['bleu'].append(None)   
                        # train loss  
                        if isfile(njoin(fpath, 'scalars', 'loss', 'train.csv')):
                            run = pd.read_csv(njoin(fpath, 'scalars', 'loss', 'train.csv'), index_col=False)                         
                            metrics_dict['train_loss'].append(run.iloc[run.index[-1],-1])                               
                        else:                            
                            metrics_dict['train_loss'].append(None)     
                        # val loss
                        if isfile(njoin(fpath, 'scalars', 'loss', 'validation.csv')):
                            run = pd.read_csv(njoin(fpath, 'scalars', 'loss', 'validation.csv'), index_col=False)                         
                            metrics_dict['val_loss'].append(run.iloc[run.index[-1],-1])                               
                        else:                            
                            metrics_dict['val_loss'].append(None)                                                                                
                        if ensembles == 1:
                            # get configs
                            f = open(njoin(fpath,'config.json'))
                            config = json.load(f)
                            f.close()
                            # config
                            for col in cols[:-len(metrics + cols_config + cols_other)] + cols_config:
                                model_dir_dct[col] = config[col]

            # metrics
            for metric in metrics:
                model_dir_dct[metric] = metrics_dict[metric]
            # others
            for col in cols_other:
                model_dir_dct[col] = locals()[col]

            df = df._append(model_dir_dct, ignore_index=True)

        DCT_ALL[model_name] = df

    return DCT_ALL

def load_model_files(model_dir):
    train_setting = pd.read_csv(njoin(model_dir, 'train_setting.csv'))
    if os.path.isdir(njoin(model_dir, 'run_performance.csv')):
        run_performance = pd.read_csv(njoin(model_dir, 'run_performance.csv'))
    else:
        run_performance = None
    if isfile(njoin(model_dir, 'train_setting.csv')):
        train_setting = pd.read_csv(njoin(model_dir, 'train_setting.csv'))
    else:
        train_setting = None
    f = open(njoin(model_dir,'config.json'))
    config = json.load(f)
    f.close()
    f = open(njoin(model_dir,'attn_setup.json'))
    attn_setup = json.load(f)
    f.close()      

    return attn_setup, config, run_performance, train_setting 

# -------------------- Main utils --------------------        

def convert_dict(dct):  # change elements of dict its value is a dict
    for key in list(dct.keys()):
        val = dct[key]
        if isinstance(val, dict):
            val_ls = list(val.values())
            assert len(val_ls) == 1, 'val_ls has len greater than one'
            dct[key] = val_ls[0]
    return dct

def convert_train_history(ls):  # convert trainer.state.log_history to df
    df_model = pd.DataFrame()
    cur_dict = convert_dict(ls[0])
    cur_step = cur_dict['step']
    for idx, next_dict in enumerate(ls[1:]):        
        next_step = next_dict['step']
        if cur_step == next_step:
            cur_dict.update(convert_dict(next_dict))
        else:
            df_model = df_model._append(cur_dict, ignore_index=True)
            cur_dict = convert_dict(next_dict)
            cur_step = cur_dict['step']
    df_model = df_model._append(cur_dict, ignore_index=True)
    return df_model