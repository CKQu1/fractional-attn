import argparse
import json
import os
import pandas as pd
from ast import literal_eval
from os.path import join, normpath, isdir, isfile

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

def get_instance(dir, *args):  # for enumerating each instance of training
    #global start, end, instances, s_part

    if isdir(dir):
        instances = []
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
                    instances.append(int(dirname[start+6:]))
                    #except:
                    #    pass       
            #print(instances)  # delete
            return max(instances) + 1 if len(instances) > 0 else 0
        else:
            return 0
    else:
        return 0

# create model_dir which determines the instance for the specific model/training setting
def create_model_dir(model_root_dir, **kwargs):
    attn = kwargs.get('attn', 'softmax')    
    task = kwargs.get('task', 'text')
    qk_share = kwargs.get('qk_share', False)

    if '-' in task:
        dataset_code = task.split('-')[1]
    else:
        dataset_code = task    
    #print(f'qk_share = {qk_share}')
    assert isinstance(qk_share,bool), f'qk_share is not bool, has value {qk_share}'

    dirname = f'{attn}-{dataset_code}'
    #dirname += '-qqv' if qk_share is True else '-qkv'  # qk weight-tying
    if 'fns' in attn:             
        manifolds_dict = {'sphere': 'sp', 'rd': 'rd'}
        manifold = kwargs.get('manifold', 'sphere')       
        if dirname[:2] not in manifolds_dict.values():
            dirname = manifolds_dict[manifold] + dirname
        else:
             assert dirname[:2] == manifolds_dict[manifold]
        alpha = kwargs.get("alpha", 1)
        bandwidth = kwargs.get("bandwidth", 1)
        a = kwargs.get("a", 1)             
        dirname += f'-alpha={alpha}-eps={bandwidth}-a={a}'
    elif attn == 'sink':
        bandwidth = kwargs.get("bandwidth", 1)     
        n_it = kwargs.get("n_it", 1)        
        dirname += f'-n_it={n_it}-eps={bandwidth}'

    models_dir = njoin(model_root_dir, task, dirname)
    # if 'instance' in kwargs:
    #     instance = kwargs.get('instance')
    # else:
    #     instance = get_instance(models_dir, 'model=')
    seed = kwargs.get('random', 0)
    model_dir = njoin(models_dir, f'seed={seed}')     
       
    return models_dir, model_dir      

# creates structural model_root based on model/training setting
def structural_model_root(**kwargs):
    
    # model config
    qk_share = str2bool(kwargs.get('qk_share', False)); n_layers = kwargs.get('n_layers')
    n_attn_heads = kwargs.get('n_attn_heads'); hidden_size = kwargs.get('hidden_size')
    # dataset
    ds = kwargs.get('dataset_name')
    # train settings
    use_custom_optim = kwargs.get('use_custom_optim')
    lr = kwargs.get('lr'); bs = kwargs.get('train_bs'); milestones = kwargs.get('milestones'); gamma = kwargs.get('gamma')
    epochs = kwargs.get('epochs')

    affix = 'qqv' if qk_share is True else 'qkv'   
    if use_custom_optim is True:   
        model_root = njoin(f'ds={ds}-layers={n_layers}-heads={n_attn_heads}-hidden={hidden_size}-epochs={epochs}-prj={affix}')                              
    else:          
        model_root = njoin(f'ds={ds}-layers={n_layers}-heads={n_attn_heads}-hidden={hidden_size}-epochs={epochs}-prj={affix}')             

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
        metrics = ['eval_loss', 'eval_accuracy']   
        metrics_dict = {}     
        for metric in metrics:
            metrics_dict[metric] = []        
        #cols_attn = ['fix_embed', 'qk_share', 'qkv_bias', 'dataset_name']
        cols_attn = ['qk_share', 'qkv_bias', 'dataset_name']
        cols_config = ['num_attention_heads', 'num_hidden_layers', 'hidden_size']
        #cols_train = ['steps_per_train_epoch']
        cols_train = []
        cols_other = ['ensembles', 'instances', 'model_dir']
        cols +=  cols_attn + metrics + cols_config + cols_train + cols_other

        df = pd.DataFrame(columns=cols)
        model_dir_dct = {}
        for model_dir in cur_model_dirs:
            ensembles = 0
            instances = []            
            for instance_dir in os.listdir(model_dir):
                if 'model=' in instance_dir:
                    fpath = njoin(model_dir, instance_dir)
                    instance = int(instance_dir.split('=')[-1])

                    # get configs
                    f = open(njoin(fpath,'config.json'))
                    config = json.load(f)
                    f.close()
                    f = open(njoin(fpath,'attn_setup.json'))
                    attn_setup = json.load(f)
                    f.close()  
                    train_setting = pd.read_csv(njoin(fpath, 'train_setting.csv'))

                    # attn-hyperparameters + attn-setup
                    for col in cols[:-len(metrics + cols_config + cols_train + cols_other)]:
                        model_dir_dct[col] = attn_setup[col]
                    # config
                    for col in cols_config:
                        model_dir_dct[col] = config[col]
                    # train setting
                    for col in cols_train:
                        model_dir_dct[col] = train_setting[col]

                    if isfile(njoin(fpath, 'run_performance.csv')):
                        ensembles += 1
                        instances.append(instance)  
                        run_perf = pd.read_csv(njoin(fpath, 'run_performance.csv'), index_col=False)                         
                        for metric in metrics:
                            metrics_dict[metric].append(run_perf.loc[run_perf.index[-1],metric])

            # metrics
            for metric in metrics:
                model_dir_dct[metric] = metrics_dict[metric]
            # others
            for col in cols_other:
                model_dir_dct[col] = locals()[col]

            df = df._append(model_dir_dct, ignore_index=True)

        DCT_ALL[model_name] = df

    return DCT_ALL


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
    df_model = pd.DataFrame ()
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