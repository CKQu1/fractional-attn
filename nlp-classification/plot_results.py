import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np
import pandas as pd

from ast import literal_eval
from os import makedirs
from os.path import isdir, isfile
from time import time
from tqdm import tqdm
from constants import *
from mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history

# ---------- Global plot settings ----------
font_type = {'family' : 'sans-serif'}
plt.rc('font', **font_type)
plt.rc('legend',fontsize=7)
#linestyles = ['solid', 'densely dashed', 'dashed', 'densely dotted', 'dotted']
linestyles = ['-', '--', '-.', ':']
#linestyles = ['-', '--', ':']
markers = ['s', 'D', 'd', 'v', '^', 'o', '.']
markersize = '3'
colors = list(mcl.TABLEAU_COLORS.keys())
# ------------------------------------------

# Plots average of metrics over ensembles
"""
python -i plot_results.py plot_ensembles .droot/formers_trained/layers=2-heads=8-hidden=768-epochs=10-qkv/
PROMPT input:
"""

def plot_ensembles(model_root_dir, metrics=['eval_loss', 'eval_accuracy'], 
                   legend=True, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global final_performance
    global model_dir, config_dict, final_metrics, ensemble_metrics, metric_plot   
    global model_dirs, subpath 

    metrics = str2ls(metrics)
    legend = str2bool(legend)
    display = str2bool(display)

    model_root_dir = model_root_dir.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(model_root_dir) if 'former' in dirname])

    # prompt to reorder file names
    for dirname_idx, dirname in enumerate(dirnames):
        for subdir in os.listdir(njoin(model_root_dir, dirname)):
            if isfile(njoin(model_root_dir, dirname, subdir, 'run_performance.csv')):
                final_performance = pd.read_csv(njoin(model_root_dir, dirname, subdir, 'final_performance.csv'))
                dataset = final_performance.loc[0,'dataset_name']
                print(f'Index {dirname_idx}: {dirname}')
                break        
    
    print(f'Dataset: {dataset}')
    print(f'model_root_dir = {model_root_dir}')

    dirname_idxs = input('Order of dirnames:')
    dirname_idxs = [int(dirname_idx) for dirname_idx in dirname_idxs.split(',')]
    assert len(dirname_idxs) <= len(dirnames), 'dirname_idxs cannot exceed dirnames'
    dirnames = [dirnames[dirname_idx] for dirname_idx in dirname_idxs]
    print(f'{metrics} \n')    

    nrows, ncols = 1, len(metrics)
    figsize = (2.5*ncols,2.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=True,sharey=False)
    #if axs.ndim == 1:
    if nrows == 1:
        if ncols > 1:
            axs = np.expand_dims(axs, axis=0)
        else:
            axs = np.expand_dims(axs, axis=[0,1])            

    # get model config
    qk_share = 'qkv' if 'qkv' in model_root_dir else 'qqv'
    config_dict = {'qk_share': qk_share}
    for ls in model_root_dir.split('/'):
        for ele in ls.split('-'):
            if '=' in ele:
                key, val = ele.split('=')
                config_dict[key] = val

    idx = 0  # axs dim 0
    model_names = []
    model_types = {}
    model_linestyles = {}
    model_markers = {}
    model_colors = {}
    N_model_types = 0        
    for jdx, dirname in enumerate(dirnames):             
        ensemble_dir = njoin(model_root_dir,dirname)
        model_dirs = []
        for subdir in os.listdir(ensemble_dir):
            subpath = njoin(ensemble_dir,subdir)
            if 'model=' in subdir and isfile(njoin(subpath,'final_performance.csv')):
                model_dirs.append(subpath.replace('\\',''))

        # get type of transformer
        df_setting = pd.read_csv(njoin(model_dirs[0],'final_performance.csv'))
        model_type = model_name = NAMES_DICT[dirname.split('-')[0]]
        if model_type not in model_types.keys():                
            model_types[model_name] = 1
            N_model_types += 1
        else:
            model_types[model_name] += 1            
        if model_name not in model_names:
            model_names.append(model_name)
        if 'fnsformer' in dirname:
            #alpha, bandwidth  = df_setting.loc[0,['alpha','bandwidth']]      
            bandwidth = df_setting.loc[0,'bandwidth']
            if 'a' in df_setting.columns:
                a = df_setting.loc[0,'a']
            else:
                a = 1
            if 'alpha' in df_setting.columns:
                alpha = df_setting.loc[0,'alpha']
            else:
                alpha = df_setting.loc[0,'beta']
            #model_settings = rf'$\alpha$ = {alpha}, $\varepsilon$ = {bandwidth}'
            model_settings = rf'$\alpha$ = {alpha}, $a$ = {a}'
            # if alpha < 2:                        
            #     d_intrinsic = df_setting.loc[0,'d_intrinsic']
            #     model_settings += rf', $d$ = {d_intrinsic}'  #$d_{\mathcal{M}}$
            model_name += f' ({model_settings})'
        elif 'sinkformer' in dirname:
            n_it  = df_setting.loc[0,'n_it']
            model_settings = rf'iter = {n_it}'
            model_name += f' ({model_settings})'
        model_linestyles[model_name] = linestyles[model_types[model_type] - 1]
        #model_linestyles[model_name] = ''
        #model_markers[model_name] = markers[model_types[model_type] - 1]
        model_colors[model_name] = colors[N_model_types - 1]

        # ensemble of training instances for the same architecture
        final_metrics = {}  # metrics of the final epoch
        count = 0
        ensemble_metrics = {}            
        for model_dir in model_dirs:
            df = pd.read_csv(njoin(model_dir, 'run_performance.csv'))     
                                
            for kdx, metric in enumerate(metrics):
                df_filtered = df[df[metric].notna()]

                if 'acc' in metric or 'f1' in metric:
                    metric_plot = df_filtered.loc[:,metric] * 100
                else:
                    metric_plot = df_filtered.loc[:,metric]
                if metric not in final_metrics.keys():
                    final_metrics[metric] = [metric_plot.iloc[-1]]
                else:
                    final_metrics[metric].append(metric_plot.iloc[-1])

                if metric not in ensemble_metrics.keys():
                    ensemble_metrics[metric] = [metric_plot]
                else:
                    ensemble_metrics[metric].append(metric_plot)

                epoch_eval_runtime = df[df['eval_runtime'].notna()].loc[:,'eval_runtime'].mean()
                train_runtime, total_flos = df_setting.loc[0,['train_runtime', 'total_flos']]
                for other_metric in ['epoch_eval_runtime', 'train_runtime', 'total_flos']:
                    if other_metric not in final_metrics.keys():
                        final_metrics[other_metric] = [locals()[other_metric]]
                    else:
                        final_metrics[other_metric].append(locals()[other_metric])
            
            count += 1
        final_metrics[f'count'] = count                        
                    
        print('-'*15)    
        print(f'{model_name} on {dataset}')
        for kdx, metric in enumerate(metrics):

            ensemble_metrics[metric] = pd.concat(ensemble_metrics[metric], axis=1).T
            # quit()

            # ----- Plots -----
            ensemble_mean = ensemble_metrics[metric].mean(0)
            ensemble_std = ensemble_metrics[metric].std(0)
            axs[idx,kdx].plot(df_filtered.loc[:,'epoch'], ensemble_mean,
                                linestyle=model_linestyles[model_name], c=model_colors[model_name],
                                #marker=model_markers[model_name], markersize=markersize,
                                label=model_name) 

            # std
            # axs[idx,kdx].fill_between(df_filtered.loc[:,'epoch'], 
            #                           ensemble_mean - ensemble_std, ensemble_mean + ensemble_std, 
            #                           color=model_colors[model_name], alpha=0.5)                                          

            if idx == 0:
                axs[idx,kdx].set_title(NAMES_DICT[metric])
            elif idx == nrows - 1:
                axs[idx,kdx].set_xlabel('Epoch')
            
            if legend:
                axs[0,0].legend(loc='upper left', #bbox_to_anchor=(0.5, 1.05),
                                ncol=1, frameon=False)    

            #axs[-1,kdx].set_ylabel(NAMES_DICT[dataset])

            # ----- Messages -----
            best = max(final_metrics[metric]) if 'acc' in metric or 'f1' in metric else min(final_metrics[metric])
            worst = min(final_metrics[metric]) if 'acc' in metric or 'f1' in metric else max(final_metrics[metric])
            median, mean = np.median(final_metrics[metric]), np.mean(final_metrics[metric])
            print(f'{metric.upper()} best, median, mean, worst: {best}, {median}, {mean}, {worst}')

        #axs[idx,0].set_ylabel(NAMES_DICT[dataset])
        print(f'Total ensembles: {ensemble_metrics[metric].shape[0]}')
        print('\n')            
        for other_metric in ['epoch_eval_runtime', 'train_runtime', 'total_flos']:
            print(f'Average total {other_metric}: {np.mean(final_metrics[other_metric])}')
        print('-'*15 + '\n')                    

    dataset_name_short = ''
    if '_' in dataset:
        for s in dataset.split('_'):
            dataset_name_short += s[0]
    else:
        dataset_name_short += dataset

    if display:
        plt.show()
    else:
        if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
        if len(config_dict.keys()) != 0:
            layers, heads, hidden = int(config_dict['layers']), int(config_dict['heads']), int(config_dict['hidden'])
            fig_file = f'prj={qk_share}-layers={layers}-heads={heads}-hidden={hidden}-'            
            fig_file += '-'.join(model_names)+'_' + f'ds={dataset_name_short}'
        else:
            fig_file = '-'.join(model_names)+'_' + f'ds={dataset_name_short}'
        if isfile(njoin(FIGS_DIR, fig_file)):
            version = len([fname for fname in os.listdir(FIGS_DIR) if fname==fig_file])
            fig_file += f'-v{version}'
        fig_file += '.pdf'
        plt.savefig(njoin(FIGS_DIR, fig_file))            
        print(f'Figure saved in {njoin(FIGS_DIR, fig_file)}')


# Plots single instance of training
"""
python -i plot_results.py plot_model .droot/formers_trained/layers\=2-heads\=8-hidden\=768-epochs\=5-qkv/\
 v3fnsformer-imdb-alpha\=1.2-eps\=1-dman=768/,v3fnsformer-imdb-alpha\=1.5-eps\=1-dman=768/,v3fnsformer-imdb-alpha\=1.8-eps\=1-dman=768/,v3fnsformer-imdb-alpha\=2.0-eps\=1/\
 0,0,0,0 imdb eval_loss,eval_accuracy  
"""
def plot_model(model_root_dir, dirnames, instances, 
               datasets, metrics, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dir, config_dict, metric_plot    
    # for local_keys in ['dirnames', 'datasets', 'metrics']:
    #     locals()[local_keys] = str2ls(locals()[local_keys])

    dirnames = str2ls(dirnames)
    datasets = str2ls(datasets)
    instances = str2ls(instances)
    metrics = str2ls(metrics)
    display = str2bool(display)

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
                    alpha, bandwidth  = df_setting.loc[0,['alpha','bandwidth']]      
                    model_settings = rf'$\alpha$ = {alpha}, $\varepsilon$ = {bandwidth}'
                    # if alpha < 2:                        
                    #     d_intrinsic = df_setting.loc[0,'d_intrinsic']
                    #     model_settings += rf', $d$ = {d_intrinsic}'  #$d_{\mathcal{M}}$
                    model_name += f' ({model_settings})'
                if 'acc' in metric or 'f1' in metric:
                    metric_plot = df_filtered.loc[:,metric] * 100
                    best_metric = metric_plot.max()
                else:
                    metric_plot = df_filtered.loc[:,metric]
                    best_metric = metric_plot.min()
                axs[idx,kdx].plot(df_filtered.loc[:,'epoch'], metric_plot,
                                  linestyle='-.', label=model_name)

                if idx == 0:
                    axs[idx,kdx].set_title(NAMES_DICT[metric])
                elif idx == nrows - 1:
                    axs[idx,kdx].set_xlabel('Epoch')

                print_metrics[metric] = [best_metric, metric_plot.iloc[-1]]  # best + final

            # ----- Messages -----            
            print('-'*15)    
            avg_eval_runtime = df[df['eval_runtime'].notna()].loc[:,'eval_runtime'].mean()
            train_runtime, total_flos = df_setting.loc[0,['train_runtime', 'total_flos']]
            print(f'{model_name} on {dataset}')
            for kdx, metric in enumerate(metrics):
                print(f'best and final {metric}: {print_metrics[metric]}')
            print(f'Total train_runtime: {train_runtime}')
            print(f'total_flos: ' + '{:.5e}'.format(total_flos))
            print(f'Average eval_runtime: {avg_eval_runtime}')
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