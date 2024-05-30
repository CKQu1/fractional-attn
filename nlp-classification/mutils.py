import os
import pandas as pd
from ast import literal_eval
from os.path import join, normpath, isdir

# -------------------- Path utils --------------------

def njoin(*args):
    return normpath(join(*args))

def str_to_bool(s):
    if isinstance(s, bool):
        return s
    else:
        literal_eval(s)

def str_to_ls(s):
    if isinstance(s, list):
        return s
    elif isinstance(s, str):
        return s.split(',')

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
    model_name = kwargs.get('model_name', 'fnsformer')    
    dataset_name = kwargs.get('dataset_name', 'imdb')
    if '_' in dataset_name:
        dataset_code = ''.join([s[0] for s in dataset_name.split('_')])
    else:
        dataset_code = dataset_name
    qk_share = kwargs.get('qk_share')
    #print(f'qk_share = {qk_share}')
    assert isinstance(qk_share,bool), f'qk_share is not bool, has value {qk_share}'

    dirname = f'{model_name}-{dataset_code}'
    #dirname += '-qqv' if qk_share is True else '-qkv'  # qk weight-tying
    if 'fnsformer' in model_name:                 
        alpha = kwargs.get("alpha", 1)
        bandwidth = kwargs.get("bandwidth", 1)
        a = kwargs.get("a", 1)             
        dirname += f'-alpha={alpha}-eps={bandwidth}-a={a}'
        # if (model_name=='v2fnsformer' or model_name=='v3fnsformer') and alpha < 2:
        #     d_intrinsic = kwargs.get('d_intrinsic')
        #     dirname += f'-dman={d_intrinsic}'
    elif model_name == 'sinkformer':
        bandwidth = kwargs.get("bandwidth", 1)     
        n_it = kwargs.get("n_it", 1)        
        dirname += f'-n_it={n_it}-eps={bandwidth}'

    models_dir = njoin(model_root_dir, dirname)
    instance = get_instance(models_dir, 'model=')
    model_dir = njoin(models_dir, f'model={instance}')         
       
    return models_dir, model_dir      

# creates structural model_root based on model/training setting
def structural_model_root(**kwargs):

    use_custom_optim = kwargs.get('use_custom_optim')
    qk_share = kwargs.get('qk_share'); n_layers = kwargs.get('n_layers')
    n_attn_heads = kwargs.get('n_attn_heads'); hidden_size = kwargs.get('hidden_size')

    lr = kwargs.get('lr'); bs = kwargs.get('bs'); milestones = kwargs.get('milestones'); gamma = kwargs.get('gamma')
    epochs = kwargs.get('epochs')

    affix = 'qqv' if qk_share==True else 'qkv'
    # if isinstance(milestones, str):
    #     milestones_str = milestones
    # else:
    #     milestones_str = ','.join(str(s) for s in milestones)    
    if use_custom_optim is True:
        # model_root = njoin(f'layers={n_layers}-heads={n_attn_heads}-hidden={hidden_size}-{affix}',
        #                    f'lr={lr}-bs={bs}-milestones={milestones_str}-gamma={gamma}-epochs={epochs}')    
        model_root = njoin(f'layers={n_layers}-heads={n_attn_heads}-hidden={hidden_size}-epochs={epochs}-{affix}')                              
    else:
        # model_root = njoin(f'layers={n_layers}-heads={n_attn_heads}-hidden={hidden_size}-{affix}',
        #                    f'lr={lr}-bs={bs}-epochs={epochs}')            
        model_root = njoin(f'layers={n_layers}-heads={n_attn_heads}-hidden={hidden_size}-epochs={epochs}-{affix}')             

    return model_root    

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
            df_model = df_model.append(cur_dict, ignore_index=True)
            cur_dict = convert_dict(next_dict)
            cur_step = cur_dict['step']
    df_model = df_model.append(cur_dict, ignore_index=True)
    return df_model