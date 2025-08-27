import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import math
import numpy as np
import pandas as pd
import regex as re

from ast import literal_eval
from itertools import product
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import NullFormatter
from os import makedirs
from os.path import isdir, isfile
from string import ascii_lowercase
from time import time
from tqdm import tqdm
from constants import *
from mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history
from mutils import collect_model_dirs, find_subdirs
from figure_utils import *

matplotlib.use("Agg")

# ---------- Global plot settings ----------
# font_type = {'family' : 'sans-serif'}
# plt.rc('font', **font_type)
# plt.rc('legend',fontsize=7)
linestyles = ['-', '--', '-.', ':']
markers = ['s', 'D', 'd', 'v', '^', 'o', '.']
markersize = '3'
#colors = list(mcl.TABLEAU_COLORS.keys())
TRANSP = 1  # transparency
# ------------------------------------------

MARKERSIZE = 4
BIGGER_SIZE = 10
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# shorten the dataset name
def shorten_dataset_name(selected_dataset):
    dataset_name_short = ''
    if isinstance(selected_dataset,str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset
    return dataset_name_short

# return median, 25/75 percentile
def get_metric_curves(run_perf_all):
    metric_m = run_perf_all.quantile(0.5,1)
    metric_l = run_perf_all.quantile(0,1)
    metric_u = run_perf_all.quantile(1,1)
    return [metric_l, metric_m, metric_u]

# aggregate all runs
def load_seed_runs(model_dir, seeds, metric):
    runs = []
    for seed in seeds:
        seed_path = njoin(model_dir, f'model={seed}')
        #print(f'seed_path: {seed_path}')
        fpath = njoin(seed_path, 'run_performance.csv')
        if not isfile(fpath):
            fpath = njoin(seed_path, '_run_performance.csv')
            if not isfile(fpath):
                continue
        run = pd.read_csv(fpath)
        epochs = run['iter'].astype(int) // int(run.loc[1,'iter'])
        if 'acc' in metric and run.loc[run.index[-1], metric] <= 1:
            run[metric] *= 100
        runs.append(run[metric])
    if len(runs)==0:
        return (None, None)
    else:
        return epochs, pd.concat(runs, axis=1)

# final epoch stats
def final_epoch_stats(run_perf_all, metric):
    epoch_index = run_perf_all.index[-1]
    metric_min = run_perf_all.loc[epoch_index,metric].min()
    metric_max = run_perf_all.loc[epoch_index,metric].max()
    metric_mid = (metric_min + metric_max) / 2

    metric_median = run_perf_all.loc[epoch_index:epoch_index+1,metric].median()
    metric_mean = run_perf_all.loc[epoch_index:epoch_index+1,metric].mean()
    metric_std = run_perf_all.loc[epoch_index:epoch_index+1,metric].std()    
    return [metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std]

# Plots average of metrics over ensembles, assumption 1 and 2 possibilities for full-sized models
"""
python -i plot_results.py plot_ensembles .droot/formers_trained/layers=2-heads=8-hidden=768-epochs=10-qkv/
"""
def phase_ensembles(models_root, selected_dataset='pathfinder-classification',
                    fns_manifold='rd', qk_share=True, selected_alphas='1.2,2',
                    metrics='val_acc,val_loss',
                    is_ops = [False,True],
                    #is_ops = [True],
                    cbar_separate=False, display=False):

    global run_perf_all, DCT_ALL, model_info, other_model_types, other_model_types, matching_df

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    qk_share, cbar_separate, display = map(str2bool, (qk_share, cbar_separate, display))
    metrics, is_ops = str2ls(metrics), str2ls(is_ops)
    # collect subdirs containing the model directories
    model_root_dirs = models_roots = find_subdirs(models_root, MODEL_SUFFIX)
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
    
    # ---- col names ----
    stats_colnames = ['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']      

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size =\
         DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,['num_heads', 'num_encoder_layers', 'hidden_size']]
    assert selected_dataset in DCT_ALL[list(DCT_ALL.keys())[0]].loc[:,'dataset_name'].unique(), 'selected_dataset does not exist'

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small       
    if selected_alphas.lower() == 'none':
        selected_alphas = alphas
    else:
        selected_alphas = [float(alpha) for alpha in str2ls(selected_alphas)]
    epss = sorted(df_model.loc[:,'bandwidth'].unique()) 
    eps = 1
    assert eps in epss, 'eps does not exist!'

    # ----- models to plot -----
    fns_model_type = fns_manifold + 'fns' + MODEL_SUFFIX    
    other_model_types = ['dp' + MODEL_SUFFIX]  # 'sink' + MODEL_SUFFIX
    model_types_to_plot = [fns_model_type] + other_model_types

    qk_shares = list(df_model.loc[:,'qk_share'].unique())    
    nrows, ncols = len(metrics), len(is_ops)     
    figsize = (3*ncols,3.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=False)  # layout='constrained'                    
    axs = matrixify_axs(axs, nrows, ncols)
    label_axs(fig, axs)  # alphabetically label subfigures

    model_types_plotted = []           
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        ax = axs[row_idx, col_idx]        
        row_stats = [] # summary statistics
        
        print(f'model_type = {model_type}')        
        for model_type in model_types_to_plot:
            if is_op:
                model_type = 'op' + model_type
            if model_type in DCT_ALL.keys():
                df_model = DCT_ALL[model_type]
            else:
                continue
            matching_df = df_model[(df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&
                                   (df_model['is_op']==is_op)&                                    
                                   (df_model['model_dir'].str.contains(selected_dataset))&
                                   (df_model['model_dir'].str.contains(f'/{model_type}-'))]

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            lstyle_model = LINESTYLE_DICT[model_type]
            for alpha in selected_alphas:
                is_fns = 'fns' in model_type
                alpha = alpha if is_fns else None
                matching_df.reset_index(drop=True, inplace=True)                
                                       
                # color
                color = HYP_CMAP(HYP_CNORM(alpha)) if is_fns else OTHER_COLORS_DICT[model_type]  
                # -------------------- SINK, DP -------------------- 
                model_info = matching_df 
                # -------------------- FNS --------------------
                if is_fns:
                    condition = (matching_df['alpha']==alpha) & (matching_df['bandwidth']==eps)
                    model_info = model_info[condition]
                
                if model_info.shape[0] > 0:
                    seeds, qk_share = (model_info[k].item() for k in ('seeds', 'qk_share'))                
                    epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)   
                else:
                    continue

                if run_perf_all is not None:
                    counter = run_perf_all.shape[1]
                    metric_curves = get_metric_curves(run_perf_all)          
                    exe_plot = ax.plot(epochs, metric_curves[1], linestyle=lstyle_model, c=color, alpha=TRANSP)          
                    if (row_idx,col_idx) == (0,0):
                        im = exe_plot                                            
                    ax.fill_between(epochs, metric_curves[0], metric_curves[2], color=color, alpha=TRANSP/2)                                                                        

                    # results of the final epoch
                    row_stats.append([model_type] + [alpha] +\
                                     final_epoch_stats(run_perf_all,metric) + [counter])    
                if not is_fns:
                    break  # only do once if model is not FNS type

        summary_stats = pd.DataFrame(data=row_stats, columns=['model_type','alpha']+stats_colnames)

        # print message
        print(metric)
        print(f'is_op = {is_op}, qk_share = {qk_share}')
        print(summary_stats)
        print('\n') 

        ax.grid()      
    # axs[0,0].set_ylim([75,85])

    # labels
    model_labels = []
    for model_type in model_types_plotted:  
        if model_type[:2] != 'op': 
            color = 'k' if 'fns' in model_type else OTHER_COLORS_DICT[model_type]            
            model_label = NAMES_DICT[model_type]
            if model_label not in model_labels:            
                axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], label=model_label)
                model_labels.append(model_label)

    # legend
    for alpha in selected_alphas[::-1]:
        axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
                      label=rf'$\alpha$ = {alpha}')           

    #ncol_legend = 2 if len(model_types_plotted) == 3 else 1
    ncol_legend = 2
    if len(model_types_plotted) >= 2:
        #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
        axs[0,0].legend(bbox_to_anchor=(0.95, 1.35),   # bbox_to_anchor=(0.85, 1.35)
                        loc='best', ncol=ncol_legend, frameon=False)                       

    #for row_idx in range(len(qk_shares)):        
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        ax = axs[row_idx, col_idx]
        #ax.set_ylabel(NAMES_DICT[metric])
        if row_idx == 0:
            #ax.set_title(NAMES_DICT[metric])
            ax_title = r'$W \in O(d)$' if is_ops[col_idx] else r'$W \notin O(d)$'
            ax.set_title(ax_title)
        
        axs[row_idx,col_idx].sharey(axs[row_idx, 0])
        axs[-1,col_idx].set_xlabel('Epochs')
        axs[row_idx,0].set_ylabel(NAMES_DICT[metrics[row_idx]])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = shorten_dataset_name(selected_dataset)
    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]
    
    SAVE_DIR = njoin(FIGS_DIR, 'lra-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'
        #fig_file += f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += f'l={num_hidden_layers}-d={hidden_size}-qk_share={qk_share}-'
        fig_file += '_'.join(model_types_short)+ '-' + metrics[0] + '-' + f'ds={dataset_name_short}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')

    # separate colorbar
    if cbar_separate:    
        """
        #fig.subplots_adjust(right=0.8)
        fig = plt.figure()
        cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])
        cbar_ticks = list(np.arange(1,2.01,0.2))
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks)
        cbar.ax.tick_params(axis='y', labelsize=tick_size)
        """
        
        fig = plt.figure()
        cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])
        cbar_ticks = list(np.linspace(1,2,6))
        
        cbar = mpl.colorbar.ColorbarBase(cbar_ax, norm=HYP_CNORM, cmap=HYP_CM)
        cbar.ax.set_yticklabels(cbar_ticks)
        cbar.ax.tick_params(axis='y', labelsize=16.5)

        plt.savefig(njoin(SAVE_DIR,"alpha_colorbar.pdf"), bbox_inches='tight')  


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])