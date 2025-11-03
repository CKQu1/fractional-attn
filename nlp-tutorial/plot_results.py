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
from UTILS.mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history
from UTILS.mutils import collect_model_dirs, find_subdirs, load_model_files
from UTILS.figure_utils import matrixify_axs, label_axs

matplotlib.use("Agg")

# ---------- Global plot settings ----------
# font_type = {'family' : 'sans-serif'}
# plt.rc('font', **font_type)
# plt.rc('legend',fontsize=7)
linestyles = ['-', '--', '-.', ':']
#linestyles = ['-', '--', ':']
markers = ['s', 'D', 'd', 'v', '^', 'o', '.']
markersize = '3'
colors = list(mcl.TABLEAU_COLORS.keys())
COLORS_ALPHA = ["#636363", "#469C76", "#2E63A6", "#C17DA5", "#C66526", "#EEE461", "#A4292F"]
# ------------------------------------------

MARKERSIZE = 4
#BIGGER_SIZE = 10
BIGGER_SIZE = 8
LEGEND_SIZE = 7
TRANSP = 1  # transparency (corresponding to alpha in plot)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# -------------------- FUNCTIONS --------------------
# return median, 25/75 percentile
def get_metric_curves(run_perf_all,lq=0.25,uq=0.75):
    metric_m = run_perf_all.quantile(0.5,1)
    metric_l = run_perf_all.quantile(lq,1)
    metric_u = run_perf_all.quantile(uq,1)
    return [metric_l, metric_m, metric_u]

# aggregate all runs
def load_seed_runs(model_dir, seeds, metric):
    runs = []
    for seed in seeds:
        seed_path = njoin(model_dir, f'model={seed}')
        fpath = njoin(seed_path, 'run_performance.csv')
        if not isfile(fpath):
            # fpath = njoin(seed_path, '_run_performance.csv')
            # if not isfile(fpath):
            #     continue
            continue
        run = pd.read_csv(fpath)
        if int(run.loc[0,'iter']) > 0:
            epochs = run['iter'].astype(int) // int(run.loc[0,'iter'])
        else:
            epochs = run['iter'].astype(int) // int(run.loc[1,'iter'])
        if 'acc' in metric and run.loc[run.index[-1], metric] <= 1:
            run[metric] *= 100
        runs.append(run[metric])
    if len(runs)==0:
        return (None, None)
    else:
        return epochs, pd.concat(runs, axis=1)

# final epoch stats
def final_epoch_stats(run_perf_all):
    metric_min = run_perf_all.tail(1).min(1).item()
    metric_max = run_perf_all.tail(1).max(1).item()
    metric_mid = (metric_min + metric_max) / 2

    metric_median = run_perf_all.tail(1).median(1).item()
    metric_mean = run_perf_all.tail(1).mean(1).item()
    metric_std = run_perf_all.tail(1).std(1).item()    
    return [metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std]
# --------------------------------------------------


# Plots average of metrics over ensembles (assumption 1 and 2 possibilities for full-sized models)
"""
python -i plot_results.py phase_ensembles .droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/
python -i plot_results.py phase_ensembles frac_attn/fractional-attn/nlp-tutorial/droot/6L-hidden=256-max_len=None-rescaled (Figure 2)
python -i plot_results.py phase_ensembles frac_attn/fractional-attn/nlp-tutorial/droot/6L-v4-hidden=256-max_len=512-rescaled (to be trained and plotted)
"""
def phase_ensembles(models_root, selected_dataset='imdb',
                    fns_manifold='rd', qk_share=False, selected_alphas='1.2,2',
                    metrics='val_acc,val_loss',
                    is_ops = [False,True],  # [False,True]
                    cbar_separate=False, display=False):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    global qk_shares, summary_stats, run_perf_all, metric_curves

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

    # isolate partiulcar setting for qk_share
    df_model = DCT_ALL[[model_type for model_type in list(DCT_ALL.keys()) if fns_manifold in model_type][0]]
    df_model.reset_index(drop=True, inplace=True)
    qk_shares = list(df_model.loc[:,'qk_share'].unique())
    print(qk_shares)
    assert qk_share in qk_shares, f'qk_share = {qk_share} setting does not exist!'
    
    # ---- col names ----
    stats_colnames = ['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']   

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,['n_heads', 'n_layers', 'hidden']]
    #dataset = DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,'dataset_name']
    assert selected_dataset in DCT_ALL[list(DCT_ALL.keys())[0]].loc[:,'dataset_name'].unique(), 'selected_dataset does not exist'

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
            
    nrows, ncols = len(metrics), len(is_ops)     
    # figsize = (3*ncols,3.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=(5,4))
    axs = matrixify_axs(axs, nrows, ncols)  # convert axs to 2D array
    # label_axs(fig, axs)  # alphabetically label subfigures             

    model_types_plotted = []
    model_types_seeds = {}     
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        ax = axs[row_idx, col_idx] 
        # summary statistics
        row_stats = []

        print(f'model_type = {model_type}')        
        for model_type in model_types_to_plot:
            if is_op:
                model_type = 'op' + model_type
            if model_type in DCT_ALL.keys():
                df_model = DCT_ALL[model_type]
            else:
                continue
            # matching conditions for model setup
            condition0 = (df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&(df_model['is_op']==is_op)&\
                         (df_model['model_dir'].str.contains(selected_dataset))&\
                         (df_model['model_dir'].str.contains(f'/{model_type}-'))
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
                    # color = 'k'
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
                    seeds, qk_share = (model_info[k].item() for k in ('seeds', 'qk_share'))                
                    epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)   
                else:
                    continue

                if run_perf_all is not None:
                    counter = run_perf_all.shape[1] - run_perf_all.tail(1).isna().sum(1).item()
                    metric_curves = get_metric_curves(run_perf_all)      
                    exe_plot = ax.plot(epochs, metric_curves[1], linestyle='-', c=color, alpha=1, clip_on=False, label='DP' if not is_fns else rf'$\alpha = {alpha}$')
                    if (row_idx,col_idx) == (0,0):
                        im = exe_plot                      
                    # Calculate std                       
                    metric_std = np.nanstd(run_perf_all.to_numpy(), axis=1)
                    ax.fill_between(epochs, metric_curves[1]-metric_std, metric_curves[1]+metric_std, color=color, alpha=0.3, clip_on=False, edgecolor='none') 

                    # results of the final epoch
                    row_stats.append([model_type, alpha] +\
                                     final_epoch_stats(run_perf_all) + [counter])    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    # ax.set_xlim([0,20])
                    # ax.set_xticks([0, 5, 10, 15, 20])
                    if row_idx == 0:
                        # ax.set_ylim(bottom=72,top=85)
                        # ax.set_yticks([75,80,85])
                        pass
                    elif row_idx == 1:
                        # ax.set_ylim([0.45, 0.6])
                        # ax.set_yticks([0.45, 0.5, 0.55])
                        pass
                if not is_fns:
                    break  # only do once if model is not FNS type

        summary_stats = pd.DataFrame(data=row_stats, columns=['model_type','alpha']+stats_colnames)

        # print message
        print(metric)
        print(f'is_op = {is_op}, qk_share = {qk_share}')
        print(summary_stats)
        print('\n')                    

    # # labels
    # model_labels = []
    # for model_type in model_types_plotted:  
    #     if model_type[:2] != 'op': 
    #         color = 'k' if 'fns' in model_type else OTHER_COLORS_DICT[model_type]            
    #         model_label = NAMES_DICT[model_type]
    #         if model_label not in model_labels:            
    #             axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], label=model_label)
    #             model_labels.append(model_label)

    # # legend
    axs[0,0].legend(loc='best', frameon=False)                     
    # for alpha in selected_alphas[::-1]:
    #     axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
    #                   label=rf'$\alpha$ = {alpha}')         
    # ncol_legend = 2  #if len(model_types_plotted) == 3 else 1
    # if len(model_types_plotted) >= 2:
    #     #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
    #     axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)                     

    # Add shared x and y labels     
    #fig.supxlabel('Epochs', fontsize='medium'); fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    for row_idx in range(len(metrics)):        
        for col_idx, is_op in enumerate(is_ops):  
            ax = axs[row_idx, col_idx]
            #ax.set_ylabel(NAMES_DICT[metric])
            if row_idx == 0:
                #ax.set_title(NAMES_DICT[metric])
                ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_ops[col_idx] else r'$\mathbf{W}_{Q,K} \notin O(d)$'
                ax.set_title(ax_title)
            
            axs[row_idx,col_idx].sharey(axs[row_idx, 0])
            axs[-1,col_idx].set_xlabel('Epochs')
        # axs[row_idx,0].set_ylabel(NAMES_DICT[metrics[row_idx]])
    axs[0,0].set_ylabel('Testing accuracy (%)')
    axs[1,0].set_ylabel('Testing loss')

    # Adjust layout
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.tight_layout()
    
    dataset_name_short = ''
    if isinstance(selected_dataset,str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'
        #fig_file += f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += f'l={num_hidden_layers}-d={hidden_size}-'
        fig_file += 'qqv-' if qk_share else 'qkv-'
        fig_file += '_'.join(model_types_short)+ '-' + metrics[0] + '-' + f'ds={dataset_name_short}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
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


# for investigatnig the effects of embedding dim and model depth
"""
python -i plot_results.py hyperparam_effects .droot/L-d-grid/
"""
def hyperparam_effects(models_root, fns_manifold='rd', is_rescale_dist=True,
                       qk_shares=[False, True], selected_alphas='1.2,2',
                       metric='val_acc', selected_dataset='imdb', depths=[1],
                       is_op=True):

    # PROCESSING
    global metric_matrix, counter_matrix, nan_counter_matrix, average_metric_matrix, run_perf_all
    global other_model_type, fns_type, layers

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', '8', 'p', 's', 'v']

    assert fns_manifold in ['sphere', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    assert metric in ['train_acc', 'train_loss', 'val_acc', 'val_loss']    
    fns_type = fns_manifold + 'fns' + MODEL_SUFFIX 
    other_model_type = 'dpformer'
    if is_op:
        fns_type = 'op' + fns_type
        other_model_type = 'op' + other_model_type
    model_types_to_plot = [fns_type, other_model_type]

    is_op, is_rescale_dist = str2bool(is_op), str2bool(is_rescale_dist)
    qk_shares = str2ls(qk_shares)        
    selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
    eps = 1

    # Regular expression pattern
    pattern = r"\d+L-hidden=\d+-max_len=512"
    if is_rescale_dist:            
        pattern += "-rescaled"

    # Extract matching subfolders
    layer_dirs_dict = {}
    layers, emb_ds = [], []
    for layer_dir in os.listdir(models_root):
        is_match = re.fullmatch(pattern, layer_dir)
        if is_match:
            #layer, emb_d = int(is_match.group(1)), int(is_match.group(2))
            layer = int(layer_dir.split('L')[0])          
            #emb_d = int(layer_dir.split('-')[1].split('=')[1])
            emb_d = int(layer_dir.split('-')[1].split('=')[1])  
            if isdir(njoin(models_root, layer_dir)):
                layer_dirs_dict[f'{layer}-{emb_d}'] = njoin(models_root, layer_dir)
            layers.append(layer)
            emb_ds.append(emb_d)
    layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 4]
    emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 65]
    
    #nrows, ncols = len(qk_shares), len(selected_alphas)
    nrows, ncols = len(qk_shares), len(layers)

    # (model_types, qk_share, L, d_model)
    N_model_types = len(selected_alphas) + 1
    average_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    std_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    max_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    min_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    average_metric_matrix[:] = np.nan
    std_metric_matrix[:] = np.nan
    max_metric_matrix[:] = np.nan
    min_metric_matrix[:] = np.nan
    counter_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    nan_counter_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    model_types_plotted = []
    for (qk_ii,qk_share),(layer_idx,layer),(emb_d_idx,emb_d) in\
         product(enumerate(qk_shares),enumerate(layers),enumerate(emb_ds)):

        qk_share_dirname = 'config_qqv' if qk_share else 'config_qkv'
        print(f'qk_share = {qk_share}, layer = {layer}, emb_d = {emb_d}')    
        # directories matching the above setting in the triple for loop
        if f'{layer}-{emb_d}' in layer_dirs_dict.keys():
            if qk_share_dirname in os.listdir(layer_dirs_dict[f'{layer}-{emb_d}']):
                setting_dir = njoin(layer_dirs_dict[f'{layer}-{emb_d}'], qk_share_dirname)
            else:
                continue
        else:
            continue
        # for _ in range(2):
        #     setting_dir = njoin(setting_dir, os.listdir(setting_dir)[0])
        setting_dir = njoin(setting_dir, 'imdb')
        setting_dir = njoin(setting_dir, os.listdir(setting_dir)[0])
        DCT_ALL = collect_model_dirs(setting_dir, suffix=MODEL_SUFFIX)
        model_df = DCT_ALL[fns_type].dropna(subset='alpha')
        model_df.reset_index(drop=True, inplace=True)

        for model_type in model_types_to_plot:
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
            for alpha_idx, alpha in enumerate(selected_alphas):  
                # if is fns type
                is_fns = 'fns' in model_type
                alpha = alpha if is_fns else None
                # -------------------- SINK, DP -------------------- 
                model_info = matching_df 
                # -------------------- FNS --------------------
                if is_fns:
                    condition = (matching_df['alpha']==alpha) & (matching_df['bandwidth']==eps)
                    model_info = model_info[condition]
                else:
                    alpha_idx = len(selected_alphas)
                
                if model_info.shape[0] > 0:
                    seeds, qk_share = (model_info[k].item() for k in ('seeds', 'qk_share'))                
                    epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)   
                else:
                    continue

                if run_perf_all is not None:
                    metric_curves = get_metric_curves(run_perf_all)  

                if run_perf_all is not None:
                    average_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmean(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                        #np.nanmedian(run_perf_all.loc[run_perf_all.index[-1]:,metric])                                                
                    std_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanstd(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                    max_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmax(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                    min_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmin(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                    counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        (~np.isnan(run_perf_all.loc[run_perf_all.index[-1]:,metric].to_numpy())).sum()                        
                    nan_counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        (np.isnan(run_perf_all.loc[run_perf_all.index[-1]:,metric].to_numpy())).sum()

                if not is_fns:
                    break  # only do once if model is NOT FNS type                            
    
    # PLOTTING (Just the two I think are most relevant)
    fig, axs = plt.subplots(1,2,figsize=(5, 2))
    
    ax = axs[0]
    ax.set_title(r'$\mathbf{Q} \neq \mathbf{K}$')

    if len(depths) == 1:
        trans = [1]
    else:
        trans = [0.5, 1]

    for aidx in range(len(selected_alphas) + 1):
        for lidx, l in enumerate(depths):
            average_metrics = average_metric_matrix[0,aidx,l-1]
            std_metrics = std_metric_matrix[0,aidx,l-1]
            is_fns = aidx < len(selected_alphas)
            # color                                 
            if is_fns:
                alpha = selected_alphas[aidx]
                legend_label = rf'$\alpha={selected_alphas[aidx]}$'
                # if (alpha == 1.2):
                #     color = '#2E63A6' if l == 2 else '#5391BF'
                # elif (alpha == 2.0):
                #     color = '#A4292F' if l == 2 else '#C86653'      
                color = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]
                #transparency = 1 - 1/(l+1)
                #transparency = l/(l+1)
                #transparency = 1 - np.exp(-l)
                transparency = trans[lidx]                     
                model_type = fns_type       
            else: 
                legend_label = 'DP'
                model_type = other_model_type
                #
                color = 'k' if l == 2 else COLORS_ALPHA[0]
                # color = 'k'
            linestyle = (0, (2,1)) if l == 2 else '-'
            X = np.array([1,2,3,4])
            if len(depths) > 1:
                legend_label = legend_label + r' $(L={})$'.format(l)
            ax.errorbar(X, average_metrics, yerr=std_metrics, 
                        fmt='.', linestyle=linestyle, 
                        label=legend_label + r' $(L=1)$', 
                        c=color, alpha=transparency, clip_on=False)

    ax = axs[1]
    ax.set_title(r'$\mathbf{Q} = \mathbf{K}$')
    average_metrics = average_metric_matrix[1,0,0]
    std_metrics = std_metric_matrix[1,0,0]
    for aidx in range(len(selected_alphas) + 1):
        for lidx, l in enumerate(depths):
            average_metrics = average_metric_matrix[1,aidx,l-1]
            std_metrics = std_metric_matrix[1,aidx,l-1]
            is_fns = aidx < len(selected_alphas)
            # color      
            if is_fns:
                alpha = selected_alphas[aidx]
                legend_label = rf'$\alpha={selected_alphas[aidx]}$'
                # if (alpha == 1.2):
                #     color = '#2E63A6' if l == 2 else '#5391BF'
                # elif (alpha == 2.0):
                #     color = '#A4292F' if l == 2 else '#C86653'   
                color = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]
                # transparency = 1 - 1/(l+1)  
                transparency = trans[lidx] 
                model_type = fns_type       
            else: 
                legend_label = 'DP'
                model_type = other_model_type
                color = 'k' if l == 2 else '#636363'
            linestyle = (0, (2,1)) if l == 2 else '-'
            X = np.array([1,2,3,4])
            if len(depths) > 1:
                legend_label = legend_label + r' $(L={})$'.format(l)
            ax.errorbar(X, average_metrics, yerr=std_metrics, 
                        fmt='.', linestyle=linestyle, label=legend_label, 
                        c=color, alpha=transparency, clip_on=False)

    for i, ax in enumerate(axs):
        ax.set_xticks(X)
        ax.set_xticklabels([8, 16, 32, 64])
        ax.set_xlabel(r'Dimension $d$')
        ax.set_ylim(top=85)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_yticks([75,80,85])
    axs[0].set_ylabel('Testing accuracy (%)')

    axs[1].legend(frameon=False, bbox_to_anchor=(1, 1))

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')    
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
    fig_file = models_root.split('/')[1] + '-'
    fig_file += 'hyperparam_effects.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')


# for plotting dynamic inference
"""
python -i plot_results.py dynamic_inference .droot/L-d-grid/
"""
def dynamic_inference(models_root, n_layer=1,
                      fns_type='fns', manifold='rd', is_rescale_dist=True, selected_alphas=[1.2, 2.0],
                      is_op=True, qk_share=False, metric='test_acc',
                      batch_size=64, is_dist_based=False):

    global model_dirs, emb_ds

    # general setting
    batch_size = int(batch_size)
    is_dist_based = str2bool(is_dist_based)    
    fname = 'dist' if is_dist_based else 'prob'
    fname += f'-bs={batch_size}-inference.csv'

    # get layers, emb_ds from regular expression
    pattern = r"\d+L-hidden=\d+-max_len=512"
    if is_rescale_dist:            
        pattern += "-rescaled"

    # Extract matching subfolders
    layer_dirs_dict = {}
    layers, emb_ds = [], []
    for layer_dir in os.listdir(models_root):
        is_match = re.fullmatch(pattern, layer_dir)
        if is_match:
            #layer, emb_d = int(is_match.group(1)), int(is_match.group(2))
            layer = int(layer_dir.split('L')[0])          
            #emb_d = int(layer_dir.split('-')[1].split('=')[1])
            emb_d = int(layer_dir.split('-')[1].split('=')[1])  
            if isdir(njoin(models_root, layer_dir)):
                layer_dirs_dict[f'{n_layer}-{emb_d}'] = njoin(models_root, layer_dir)
            layers.append(layer)
            emb_ds.append(emb_d)
    layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 4]
    emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 65]    
    assert n_layer in layers, f'{n_layer} does not exist!'

    # get all model dirs
    pattern = re.compile(r"model=\d+$")  # seed paths
    all_model_dirs = [str(p) for p in Path(models_root).rglob("*") if p.is_dir() and pattern.search(str(p))]    
    model_dirs = []
    fns_type = manifold + 'fns' + MODEL_SUFFIX
    other_type = 'dp'+MODEL_SUFFIX
    if is_op:
        fns_type = 'op' + fns_type
        other_type = 'op' + other_type
    model_types_to_plot = [fns_type, other_type]
    for model_dir in all_model_dirs:
        is_fns = f'/{fns_type}' in model_dir
        is_dp = f'/{other_type}' in model_dir
        if is_fns:
            for alpha in selected_alphas:
                if f'alpha={float(alpha)}' in model_dir:
                    if model_dir is not None and isfile(njoin(model_dir, fname)):
                        model_dirs.append(model_dir)
        elif is_dp:
            if model_dir is not None and isfile(njoin(model_dir, fname)):
                model_dirs.append(model_dir)

    # number of controlled variables
    inference = pd.read_csv(njoin(model_dirs[0], fname))
    controlled_vars = inference.loc[:,'controlled_variable']  # either distance based or probability based
    N_control_var = len(controlled_vars)
    ensembles = 5  # figure out how to extract this

    metrics_dynamic = np.zeros([len(selected_alphas)+1, 1, 
                                len(emb_ds), N_control_var, ensembles])
    metrics_dynamic[:] = np.nan
    for model_dir in model_dirs:
        # load config
        attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
        if attn_setup['qk_share'] == qk_share:
            seed, model_name = attn_setup['seed'], attn_setup['model_name']
            hidden = config['hidden']
            is_fns = model_name[-9:] == 'fns' + MODEL_SUFFIX
            if is_fns:
                alpha = attn_setup['alpha']
                alpha_idx = selected_alphas.index(alpha)
            else:
                alpha_idx = len(selected_alphas)
            inference = pd.read_csv(njoin(model_dir, fname))
            metrics_dynamic[alpha_idx, 0, list(emb_ds).index(hidden), :, seed] =\
                inference.loc[:,metric]

    # PLOTTING
    fig, axs = plt.subplots(1,4,figsize=(6,1.7))
    
    for didx, alpha_idx in\
          product(range(len(emb_ds)), range(len(selected_alphas)+1)):
        is_fns = alpha_idx < len(selected_alphas)
        if is_fns:
            alpha = selected_alphas[alpha_idx]
            # color = HYP_CMAP(HYP_CNORM(alpha))
            color = '#2E63A6' if alpha == 1.2 else '#A4292F'
        else:
            # color = OTHER_COLORS_DICT[other_type]
            # color = 'k'
            color = '#636363'

        metric_mean = np.nanmean(metrics_dynamic[alpha_idx,0,didx,:,:],-1)
        metric_std = np.nanstd(metrics_dynamic[alpha_idx,0,didx,:,:],-1)
              
        if didx == 0:
            axs[didx].plot(controlled_vars, metric_mean,
                                markersize=MARKERSIZE, label=rf'$\alpha$ = {alpha}' if is_fns else 'DP',
                                c=color, linestyle=LINESTYLE_DICT[fns_type])  
        else:
            axs[didx].plot(controlled_vars, metric_mean,
                                markersize=MARKERSIZE, 
                                c=color, linestyle=LINESTYLE_DICT[fns_type])  
        # axs[row, col].errorbar(controlled_vars, metric_mean, yerr=metric_std, fmt='.',
        #                     c=color, linestyle=LINESTYLE_DICT[fns_type])  

        # # error bars
        axs[didx].fill_between(controlled_vars,  metric_mean - metric_std, metric_mean + metric_std,
                                    color=color, alpha=0.2, edgecolor='none')                           

        axs[didx].spines['top'].set_visible(False)
        axs[didx].spines['right'].set_visible(False)
        axs[didx].set_xticks([0,0.5,1])
        axs[didx].set_yticks([0.5,0.7,0.9])
        axs[didx].set_xlim([0,1])
        axs[didx].set_ylim([0.5,0.9])
        axs[didx].set_xlabel(r'$p$')
        axs[didx].set_title(rf'$d = {emb_ds[didx]}$')

    # legends
    # for alpha_idx, alpha in enumerate(selected_alphas):
    #     c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
    #     axs[0,0].plot([], [], c=c_hyp, linestyle=LINESTYLE_DICT[fns_type],
    #                 label=rf'$\alpha$ = {alpha}')    
    # axs[0,0].plot([],[], c=OTHER_COLORS_DICT[other_type],linestyle=LINESTYLE_DICT[other_type])                      
    fig.legend(frameon=False, bbox_to_anchor=(0.75,0.1), ncol=3)
                        
    # control_var_name = 'Distance threshold' if is_dist_based else 'Removal probability'
    # for col in range(2):
    #     axs[0].set_title(rf'$d = {emb_ds[col]}$')
        # axs[0].set_xlabel(control_var_name)
    axs[0].set_ylabel('Testing accuracy (%)')

    # abbreviate dataset_name
    dataset = attn_setup['dataset_name']
    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    
    qkv = 'qqv' if qk_share else 'qkv'
    fig_file = f'{n_layer}L-{metric}-'
    if is_dist_based:           
        fig_file += f'dynamic_inference_dist'
    else:
        fig_file += f'dynamic_inference_prob'
    fig_file += f'-{qkv}.pdf'

    plt.tight_layout()
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')        


"""
python -i plot_results.py len_inference .droot/L-d-grid/
"""
def len_inference(models_root, n_layer=6, max_len_adj=1024,
                  fns_type='fns', manifold='rd', is_rescale_dist=True, selected_alphas=[1.2, 2.0],
                  is_op=False, qk_shares=[False], metric='test_acc'):

    global model_dirs, emb_ds, metric_plot, metrics_all, inference, layers, emb_ds

    # general setting
    if metric == 'test_acc':
        fname = f'test_inference-bs=1-len={max_len_adj}.csv'
    elif metric == 'train_acc':
        fname = f'train_inference-bs=1-len={max_len_adj}.csv'

    # get layers, emb_ds from regular expression
    pattern = r"\d+L-hidden=\d+-max_len=512"
    if is_rescale_dist:            
        pattern += "-rescaled"

    # Extract matching subfolders
    layer_dirs_dict = {}
    layers, emb_ds = [], []
    for layer_dir in os.listdir(models_root):
        is_match = re.fullmatch(pattern, layer_dir)
        if is_match:
            #layer, emb_d = int(is_match.group(1)), int(is_match.group(2))
            layer = int(layer_dir.split('L')[0])          
            #emb_d = int(layer_dir.split('-')[1].split('=')[1])
            emb_d = int(layer_dir.split('-')[1].split('=')[1])  
            if isdir(njoin(models_root, layer_dir)):
                layer_dirs_dict[f'{n_layer}-{emb_d}'] = njoin(models_root, layer_dir)
            layers.append(layer)
            emb_ds.append(emb_d)
    layers = np.array(sorted(list(set(layers)))); # layers = layers[layers < 4]
    emb_ds = np.array(sorted(list(set(emb_ds)))); # emb_ds = emb_ds[emb_ds < 65]    
    assert n_layer in layers, f'{n_layer} does not exist!'

    # get all model dirs
    pattern = re.compile(r"model=\d+$")  # seed paths
    all_model_dirs = [str(p) for p in Path(models_root).rglob("*") if p.is_dir() and pattern.search(str(p))]    
    model_dirs = []
    fns_type = manifold + 'fns' + MODEL_SUFFIX
    other_type = 'dp'+MODEL_SUFFIX
    if is_op:
        fns_type = 'op' + fns_type
        other_type = 'op' + other_type
    model_types_to_plot = [fns_type, other_type]
    for model_dir in all_model_dirs:
        is_fns = f'/{fns_type}' in model_dir
        is_dp = f'/{other_type}' in model_dir
        if is_fns:
            # isolate alphas from SELECTED_ALPHAS
            if not any(f'alpha={float(alpha)}' in model_dir for alpha in selected_alphas):
                continue     
        elif is_dp:
            pass
        else:
            continue
        # elif is_dp:
        if model_dir is not None and isfile(njoin(model_dir, fname)):
            model_dirs.append(model_dir)

    # number of controlled variables
    inference = pd.read_csv(njoin(model_dirs[0], fname))
    seq_lens = inference.loc[:,'seq_len']
    _, config, _, _ = load_model_files(model_dir)    
    thresholds = []
    ii = 6
    #while 2**ii <= config['seq_len']:
    while 2**ii <= max_len_adj:
        thresholds.append(2**ii)
        ii += 1    
    # if seq_lens.max() > max_len_adj:
    #     thresholds.append(seq_lens.max())

    ensembles = 5  # figure out how to extract this

    nrows, ncols = len(qk_shares), len(emb_ds)
    figsize = (3.65*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'
    axs = matrixify_axs(axs, nrows, ncols)
    label_axs(fig, axs)

    metrics_all = np.zeros([2, len(selected_alphas)+1, len(qk_shares), 
                               len(emb_ds), len(thresholds) + 1, ensembles])
    metrics_all[:] = np.nan
    for model_dir in model_dirs:
        # load config
        attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
        seed, model_name, qk_share = attn_setup['seed'], attn_setup['model_name'],\
              attn_setup['qk_share']
        hidden = config['hidden']
        is_fns = model_name[-9:] == 'fns' + MODEL_SUFFIX
        if is_fns:
            alpha = attn_setup['alpha']
            alpha_idx = selected_alphas.index(alpha)
        else:
            alpha_idx = len(selected_alphas)
        #if isfile(njoin(model_dir, fname)):
        inference = pd.read_csv(njoin(model_dir, fname))
        for tidx, threshold in enumerate(thresholds):
            if tidx == 0:
                mask = inference["seq_len"] <= threshold
            else:
                mask = (thresholds[tidx-1] < inference["seq_len"]) & (inference["seq_len"] <= threshold)
            metrics_all[:, alpha_idx, qk_shares.index(qk_share), list(emb_ds).index(hidden), tidx, seed] =\
                [inference.loc[mask, "is_correct"].sum(), mask.sum()]                

        # greater than max_len
        mask = (thresholds[-1] < inference["seq_len"])
        metrics_all[:, alpha_idx, qk_shares.index(qk_share), list(emb_ds).index(hidden), -1, seed] =\
            [inference.loc[mask, "is_correct"].sum(), mask.sum()]                

    # accuracy is count / total
    metric_plot = metrics_all[0,:] / metrics_all[1,:]
    for sidx, didx, alpha_idx in\
          product(range(len(qk_shares)), range(len(emb_ds)), range(len(selected_alphas)+1)):
        
        ax = axs[sidx,didx]

        is_fns = alpha_idx < len(selected_alphas)
        if is_fns:
            alpha = selected_alphas[alpha_idx]
            color = HYP_CMAP(HYP_CNORM(alpha))
        else:
            color = OTHER_COLORS_DICT[other_type]

        metric_mean = np.nanmean(metric_plot[alpha_idx,sidx,didx,:,:] * 100,-1)
        metric_std = np.nanstd(metric_plot[alpha_idx,sidx,didx,:,:]  * 100,-1)
                            
        # add final dummy threshold
        ax.plot(thresholds + [thresholds[-1] * 2], metric_mean,
                            markersize=MARKERSIZE,
                            c=color, linestyle=LINESTYLE_DICT[fns_type])  

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # error bars
        # ax.fill_between(thresholds + [thresholds[-1] * 2],  
        #                 metric_mean - metric_std, metric_mean + metric_std,
        #                 color=color, alpha=1/2)                           

    # legends
    for alpha_idx, alpha in enumerate(selected_alphas):
        c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
        axs[0,0].plot([], [], c=c_hyp, linestyle=LINESTYLE_DICT[fns_type],
                    label=rf'$\alpha$ = {alpha}')    
    axs[0,0].plot([],[], c=OTHER_COLORS_DICT[other_type],linestyle=LINESTYLE_DICT[other_type])                      
                            
    # log x-axis
    axs[0,0].set_xscale('log')                            

    for ncol in range(ncols):
        axs[0,ncol].set_title(rf'$d = {emb_ds[ncol]}$')
        axs[-1,ncol].set_xlabel('Sequence length')        
        # tick labels
        axs[-1,ncol].set_xticks(thresholds + [thresholds[-1] * 2])
        axs[-1,ncol].set_xticklabels(thresholds + [rf'$\leq$'])  # [rf'${thresholds[-1]} \leq$']
        # remove minor ticks
        axs[-1,ncol].xaxis.set_minor_formatter(NullFormatter()) 
        axs[-1,ncol].xaxis.minorticks_off() 
    for nrow in range(nrows):
        axs[nrow,0].set_ylabel(r'$Q = K$' if qk_shares[nrow] else r'$Q \neq K$')
    axs[0,0].legend(frameon=False)
    
    plt.tight_layout(rect=[0, 0, 0.93, 1])   

    # abbreviate dataset_name
    dataset = attn_setup['dataset_name']
    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    
    qkv = 'qqv' if qk_share else 'qkv'
    fig_file = f'{n_layer}L-len={max_len_adj}-is_op={is_op}-{metric}-inference.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')  


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])