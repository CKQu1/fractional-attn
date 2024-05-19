import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ast import literal_eval
from os import makedirs
from os.path import isdir, isfile
from time import time
from tqdm import tqdm
from constants import *
from mutils import njoin, str_to_bool, str_to_ls, create_model_dir, convert_train_history

# ---------- Global plot settings ----------
font_type = {'family' : 'sans-serif'}
plt.rc('font', **font_type)
plt.rc('legend',fontsize=7)
# ------------------------------------------

# Example:
"""
python -i plot_results.py plot_model .droot/trained_models_v6\
 v3fnsformer-imdb-qqv-beta=1.0-eps=1-dman=5,v3fnsformer-imdb-qqv-beta=2.0-eps=1,dpformer-imdb-qqv\
 0,0,0 imdb eval_loss,eval_accuracy
"""
def plot_model(model_root_dir, dirnames, instances, 
               datasets, metrics, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dir, config_dict, metric_plot    
    # for local_keys in ['dirnames', 'datasets', 'metrics']:
    #     locals()[local_keys] = str_to_ls(locals()[local_keys])

    dirnames = str_to_ls(dirnames)
    datasets = str_to_ls(datasets)
    instances = str_to_ls(instances)
    metrics = str_to_ls(metrics)
    display = str_to_bool(display)

    model_root_dir = model_root_dir.replace('\\','')
    print(f'model_root_dir = {model_root_dir}')
    print(f'{metrics} \n')

    nrows, ncols = len(datasets), len(metrics)
    figsize = (10,2.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=True,sharey=False)
    if axs.ndim == 1:
        axs = np.expand_dims(axs, axis=0)

    # get model config
    qk_share = 'qkv' if 'qkv' in model_root_dir else 'qqv'
    config_dict = {}
    for ls in model_root_dir.split('/'):
        for ele in ls.split('-'):
            if '=' in ele:
                key, val = ele.split('=')
                config_dict[key] = val

    #quit()  # delete
    model_names = []
    for idx, dataset in tqdm(enumerate(datasets)):
        for jdx, dirname in enumerate(dirnames): 
            model_dir = njoin(model_root_dir, dirname, f'model={instances[jdx]}')
            model_dir = model_dir.replace('\\','')
            df = pd.read_csv(njoin(model_dir, 'run_performance.csv'))    
            df_setting = pd.read_csv(njoin(model_dir,'final_performance.csv'))   
            print_metrics = {}     
            for kdx, metric in enumerate(metrics):
                df_filtered = df[df[metric].notna()]

                model_name = NAMES_DICT[dirname.split('-')[0]]
                if model_name not in model_names:
                    model_names.append(model_name)
                if 'fnsformer' in dirname:
                    beta, bandwidth  = df_setting.loc[0,['beta','bandwidth']]      
                    model_settings = rf'$\beta$ = {beta}, $\varepsilon$ = {bandwidth}'
                    # if beta < 2:                        
                    #     d_intrinsic = df_setting.loc[0,'d_intrinsic']
                    #     model_settings += rf', $d$ = {d_intrinsic}'  #$d_{\mathcal{M}}$
                    model_name += f' ({model_settings})'
                if 'acc' in metric or 'f1' in metric:
                    metric_plot = df_filtered.loc[:,metric] * 100
                    best_metric = metric_plot.max()
                else:
                    metric_plot = df_filtered.loc[:,metric]
                    best_metric = metric_plot.min()
                axs[idx,kdx].plot(df_filtered.loc[:,'epoch'], metric_plot, label=model_name)

                if idx == 0:
                    axs[idx,kdx].set_title(NAMES_DICT[metric])
                elif idx == nrows - 1:
                    axs[idx,kdx].set_xlabel('Epoch')

                print_metrics[metric] = [best_metric, metric_plot.iloc[-1]]  # best + final

            # ----- Messages -----            
            print('-'*15)    
            print(f'{model_name} on {dataset}')
            for kdx, metric in enumerate(metrics):
                print(f'best and final {metric}: {print_metrics[metric]}')
            print('-'*15 + '\n')                    

        axs[idx,0].set_ylabel(NAMES_DICT[dataset])
    #axs[0,0].legend(loc=7)
    axs[0,0].legend(loc='upper left', #bbox_to_anchor=(0.5, 1.05),
                    ncol=1, frameon=False)

    if display:
        plt.show()
    else:
        if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
        layers, heads, hidden = int(config_dict['layers']), int(config_dict['heads']), int(config_dict['hidden'])
        fig_file = f'layers={layers}-heads={heads}-hidden={hidden}-'
        fig_file += '-'.join(model_names)+'_'+'-'.join(datasets)
        if isfile(njoin(FIGS_DIR, fig_file)):
            version = len([fname for fname in os.listdir(FIGS_DIR) if fname==fig_file])
            fig_file += f'-v{version}'
        fig_file += '.pdf'
        plt.savefig(njoin(FIGS_DIR, fig_file))            
        print(f'Figure saved in {njoin(FIGS_DIR, fig_file)}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])