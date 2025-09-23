import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import math
import numpy as np
import pandas as pd
import re

from ast import literal_eval
from itertools import product
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import NullFormatter
from os import makedirs
from os.path import isdir, isfile
from pathlib import Path
from string import ascii_lowercase
from time import time
from tqdm import tqdm
from constants import *
from utils.mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history
from utils.mutils import collect_model_dirs, find_subdirs

# -------------------- FUNCTIONS --------------------
# return median, 25/75 percentile
def get_metric_curves(run_perf_all,lq=0.25,uq=0.75):
    metric_m = run_perf_all.quantile(0.5,1)
    metric_l = run_perf_all.quantile(lq,1)
    metric_u = run_perf_all.quantile(uq,1)
    return [metric_l, metric_m, metric_u]

# aggregate all runs
def load_seed_runs(model_dir, seeds, metric):
    assert metric in ['bleu', 'train_loss', 'val_loss'], f'metric = {metric} does not exist!'
    runs = []
    for seed in seeds:
        seed_path = njoin(model_dir, f'model={seed}')
        dirname = njoin(seed_path, 'scalars')
        # bleu score
        if metric == 'bleu':
            fpath = njoin(dirname, 'bleu.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'bleu'])
            run.iloc[:,-1] = run.iloc[:,-1] * 100
        # train loss
        elif metric == 'train_loss':
            fpath = njoin(dirname, 'loss', 'train.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'train_loss'])       
        # test loss
        elif metric == 'val_loss':
            fpath = njoin(dirname, 'loss', 'validation.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'val_loss'])   
        else:
            run = None
        if run is not None:
            runs.append(run.iloc[:,1])
            epochs = run.iloc[:,0]
    if len(runs)==0:
        return (None, None)
    else:
        return epochs, pd.concat(runs, axis=1)

# final epoch stats
def final_epoch_stats(runs):
    epoch_index = runs.index[-1]
    metric_min = runs.loc[epoch_index,:].min()
    metric_max = runs.loc[epoch_index,:].max()
    metric_mid = (metric_min + metric_max) / 2

    metric_median = runs.loc[epoch_index:epoch_index+1,:].median(1).item()
    metric_mean = runs.loc[epoch_index:epoch_index+1,:].mean(1).item()
    metric_std = runs.loc[epoch_index:epoch_index+1,:].std(1).item()    
    return [metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std]
# --------------------------------------------------
def matrixify_axs(axs, nrows, ncols):
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)   
    return axs

def phase_ensembles(models_root, selected_dataset='en-de',
                    fns_manifold='rd', selected_alphas='1.2,2',
                    metrics='bleu,train_loss',  # train_loss, val_loss
                    is_ops = [False,True],  # [False,True]
                    cbar_separate=False, display=False):

    global DCT_ALL, model_root_dirs, df_model, run_perf_all, matching_df
    global model_info, seeds, metric_curves, epochs, metric_std

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    cbar_separate, display = map(str2bool, (cbar_separate, display))
    metrics, is_ops = str2ls(metrics), str2ls(is_ops)

    # collect subdirs containing the model directories
    model_root_dirs = models_roots = find_subdirs(njoin(models_root), MODEL_SUFFIX)
    print(model_root_dirs)                  

    # all trained model types
    model_types = []   
    DCT_ALL = {} 
    for model_root_dir in model_root_dirs:
        DCT_cur = collect_model_dirs(model_root_dir, suffix=MODEL_SUFFIX)
        for model_type, df_model_cur in DCT_cur.items():
            df_clean = df_model_cur.dropna(subset='alpha') if 'alpha' in df_model_cur.columns else df_model_cur
            if model_type not in DCT_ALL:
                model_types.append(model_type)
                DCT_ALL[model_type] = df_clean
            else:
                DCT_ALL[model_type] = pd.concat([DCT_ALL[model_type], df_clean], ignore_index=True)               

    df_model = DCT_ALL[[model_type for model_type in list(DCT_ALL.keys()) if fns_manifold in model_type][0]]
    df_model.reset_index(drop=True, inplace=True)
    
    # print('df_model')
    # print(df_model)

    # ---- col names ----
    stats_colnames = ['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']   

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    epss = sorted(df_model.loc[:,'bandwidth'].unique())    
    if selected_alphas.lower() == 'none':
        selected_alphas = alphas
    else:
        selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
    #eps = epss[0]
    eps = 1  # hard coded

    # ----- models to plot -----
    fns_model_type = fns_manifold + 'fns' + MODEL_SUFFIX    
    other_model_types = ['dp' + MODEL_SUFFIX]  # 'sink' + MODEL_SUFFIX
    model_types_to_plot = [fns_model_type] + other_model_types
            
    print(f'model_types_to_plot: {model_types_to_plot}')

    nrows, ncols = len(metrics), len(is_ops)     
    figsize = (2.5*len(is_ops),2*len(metrics))
    #figsize = (1,1)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True)
    axs = matrixify_axs(axs, nrows, ncols)  # convert axs to 2D array
    # label_axs(fig, axs)  # alphabetically label subfigures             

    model_types_plotted = []
    model_types_seeds = {}     
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        ax = axs[row_idx, col_idx] 
        # summary statistics
        row_stats = []

        #print(f'model_type = {model_type}')        
        for model_type in model_types_to_plot:
            if is_op:
                model_type = 'op' + model_type
            if model_type in DCT_ALL.keys():
                df_model = DCT_ALL[model_type]
            else:
                continue
            # matching conditions for model setup
            condition0 = (df_model['ensembles']>0)&(df_model['is_op']==is_op)&\
                         (df_model['model_dir'].str.contains(selected_dataset))&\
                         (df_model['model_dir'].str.contains(f'{model_type}-'))   
            matching_df = df_model[condition0]

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            lstyle_model = LINESTYLE_DICT[model_type]
            for alpha in selected_alphas:
                is_fns = 'fns' in model_type
                alpha = alpha if is_fns else None
                matching_df.reset_index(drop=True, inplace=True)                                  

                # color
                if is_fns:
                    color = '#2E63A6' if alpha == 1.2 else '#A4292F'
                else:
                    color = '#636363'
                # color = HYP_CMAP(HYP_CNORM(alpha)) if is_fns else OTHER_COLORS_DICT[model_type]  
                # -------------------- SINK, DP -------------------- 
                model_info = matching_df 
                # -------------------- FNS --------------------
                if is_fns:
                    # matching conditions for FNS setup
                    condition = (matching_df['alpha']==alpha) & (matching_df['bandwidth']==eps)
                    model_info = model_info[condition]
                # get aggregated training curves
                if model_info.shape[0] > 0:
                    seeds = model_info['seeds'].item()                
                    epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)                       
                else:
                    continue

                #EPOCHS_PLOT = 49
                if run_perf_all is not None:
                    counter = run_perf_all.shape[1]
                    metric_curves = get_metric_curves(run_perf_all)      
                    exe_plot = ax.plot(epochs + 1, metric_curves[1], linestyle='-', c=color, alpha=1, clip_on=False, label='DP' if not is_fns else rf'$\alpha = {alpha}$')
                    if (row_idx,col_idx) == (0,0):
                        im = exe_plot                      
                    # Calculate std                       
                    metric_std = np.nanstd(run_perf_all.to_numpy(), axis=1)
                    ax.fill_between(epochs + 1, metric_curves[1]-metric_std, metric_curves[1]+metric_std, color=color, alpha=0.3, clip_on=False, edgecolor='none') 

                    # results of the final epoch
                    row_stats.append([model_type, alpha] +\
                                     final_epoch_stats(run_perf_all) + [counter])    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlim([0,epochs.iloc[-1].item() + 1])
                    #ax.set_xticks([0] + list(range(25,126,25)))
                if not is_fns:
                    break  # only do once if model is not FNS type

        summary_stats = pd.DataFrame(data=row_stats, columns=['model_type','alpha']+stats_colnames)

        # print message
        print(metric)
        print(f'is_op = {is_op}')
        print(summary_stats)
        print('\n')                    

    for cidx in range(ncols):
        axs[0,cidx].set_ylim(0,35)
        axs[0,cidx].margins(0)
        #ax.set_yticks([70,75,80])
        # axs[1,cidx].set_ylim([0.6,2.1])
        # axs[1,cidx].set_yticks([1, 1.5, 2])
        axs[1,cidx].margins(0)

    # legend
    axs[0,0].legend(loc='best', frameon=False)                     

    for row_idx in range(nrows):
        for col_idx, is_op in enumerate(is_ops):  
            ax = axs[row_idx, col_idx]
            if row_idx == 0:                
                ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_ops[col_idx] else r'$\mathbf{W}_{Q,K} \notin O(d)$'
                ax.set_title(ax_title)            
            axs[row_idx,col_idx].sharey(axs[row_idx, 0])
            axs[-1,col_idx].set_xlabel('Epochs')
    # axs[row_idx,0].set_ylabel(NAMES_DICT[metrics[row_idx]])
    axs[0,0].set_ylabel('Bleu score (%)')
    if metrics[1] == 'val_loss':
        axs[1,0].set_ylabel('Testing loss')
    elif metrics[1] == 'train_loss':
        axs[1,0].set_ylabel('Training loss')

    # Adjust layout
    plt.subplots_adjust(wspace=0.4, hspace=0.3)            

    dataset_name_short = ''
    if isinstance(selected_dataset,str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]
    
    #return fig, axs

    plt.tight_layout()
    SAVE_DIR = njoin(FIGS_DIR, 'translation-task')   
    if not isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR) 
    fig_file = models_root.split('/')[1] + '-' + 'phase_ensembles'
    fig_file += '.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')
    # plt.show()    


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
