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
                is_append = (len(os.listdir(njoin(dir, dirname))) > 0)  # make sure file is non-empty
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

def create_model_dir(model_root_dir, **kwargs):
    model_name = kwargs.get('model_name', 'fnsformer')    
    dataset_name = kwargs.get('dataset_name', 'imdb')    
          
    models_dir = njoin(model_root_dir, f"{model_name}_{dataset_name}")
    if not os.path.isdir(models_dir): os.makedirs(models_dir)   
    if model_name == 'fnsformer': 
        beta = kwargs.get("beta", 1)
        bandwidth = kwargs.get("bandwidth", 1)        
        instance = get_instance(models_dir, 'model=', f'_beta={beta}_eps={bandwidth}')
        model_dir = njoin(models_dir, f"model={instance}_beta={beta}_eps={bandwidth}")
    else:
        instance = get_instance(models_dir, 'model=')
        model_dir = njoin(models_dir, f'model={instance}')        
    if not os.path.isdir(model_dir): os.makedirs(model_dir)     
       
    return models_dir, model_dir      

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