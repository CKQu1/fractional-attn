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
# return mean and mean +/- sample standard deviation
def get_metric_curves(run_perf_all, type='median'):
    if type == 'mean':
        metric_mean = run_perf_all.mean(axis=1)
        metric_std = run_perf_all.std(axis=1, ddof=1)
        metric_l = metric_mean - metric_std
        metric_u = metric_mean + metric_std
    elif type == 'median':
        metric_mean = run_perf_all.median(axis=1)
        metric_l = run_perf_all.quantile(0.25, axis=1)
        metric_u = run_perf_all.quantile(0.75, axis=1)

    return [metric_l, metric_mean, metric_u]

# aggregate all runs
def load_seed_runs(model_dir, seeds, metric, comparison_bleu_methods=None):
    assert metric in ['bleu', 'train_loss', 'val_loss', 'lr'], f'metric = {metric} does not exist!'
    runs = []
    bleu_methods = set()
    for seed in seeds:
        seed_path = njoin(model_dir, f'model={seed}')
        dirname = njoin(seed_path, 'scalars')
        # bleu score
        if metric == 'bleu':
            fpath = njoin(dirname, 'bleu.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'bleu'])
            # Legacy NLTK runs stored BLEU on 0-1; SacreBLEU uses 0-100.
            train_config_path = njoin(seed_path, 'train_config.json')
            bleu_metadata = {}
            if isfile(train_config_path):
                with open(train_config_path) as config_file:
                    bleu_metadata = json.load(config_file)
            bleu_scale = bleu_metadata.get('bleu_scale')
            bleu_methods.add(
                bleu_metadata.get('bleu_method', 'legacy_teacher_forced_sentence_bleu')
            )
            if (
                bleu_scale != '0-100'
                and not run['bleu'].dropna().empty
                and run['bleu'].abs().max() <= 1
            ):
                run.loc[:, 'bleu'] = run['bleu'] * 100
        # train loss
        elif metric == 'train_loss':
            fpath = njoin(dirname, 'loss', 'train.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'train_loss'])       
        # validation loss
        elif metric == 'val_loss':
            fpath = njoin(dirname, 'loss', 'validation.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'val_loss'])   
        # learning rate
        elif metric == 'lr':
            fpath = njoin(dirname, 'lr.csv')
            if not isfile(fpath):
                continue
            run = pd.read_csv(fpath, header=None, index_col=False, names=['epoch', 'lr'])
        else:
            run = None
        if run is not None and not run.empty:
            if run['epoch'].duplicated().any():
                raise ValueError(f'Duplicate epochs found in {fpath}')
            metric_values = run.set_index('epoch').iloc[:,0]
            metric_values.name = seed
            runs.append(metric_values)
    if len(runs)==0:
        return (None, None)
    elif metric == 'bleu' and len(bleu_methods) != 1:
        raise ValueError(f'Cannot aggregate different BLEU methods: {sorted(bleu_methods)}')
    else:
        if metric == 'bleu' and comparison_bleu_methods is not None:
            comparison_bleu_methods.update(bleu_methods)
            if len(comparison_bleu_methods) != 1:
                raise ValueError(
                    'Cannot compare different BLEU methods: '
                    f'{sorted(comparison_bleu_methods)}'
                )
        run_perf_all = pd.concat(runs, axis=1).sort_index().dropna(how='all')
        if run_perf_all.empty:
            return (None, None)
        epochs = pd.Series(run_perf_all.index.to_numpy(), name='epoch')
        return epochs, run_perf_all

# final epoch stats
def final_epoch_stats(runs):
    final_values = runs.iloc[-1,:].dropna()
    metric_min = final_values.min()
    metric_max = final_values.max()
    metric_mid = (metric_min + metric_max) / 2

    metric_median = final_values.median()
    metric_mean = final_values.mean()
    metric_std = final_values.std(ddof=1)
    counter = final_values.count()
    return [metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std, counter]
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
                    metrics='bleu,val_loss',  # bleu,train_loss lr
                    is_ops = [False,True],  # [False,True]
                    cbar_separate=False, display=False):

    global DCT_ALL, model_root_dirs, df_model, run_perf_all, matching_df
    global model_info, seeds, metric_curves, epochs, metric_std

    start_epoch = 15

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
    other_model_types = ['dp' + MODEL_SUFFIX, 'sink' + MODEL_SUFFIX]  # 'sink' + MODEL_SUFFIX
    model_types_to_plot = [fns_model_type] + other_model_types
            
    print(f'model_types_to_plot: {model_types_to_plot}')

    nrows, ncols = len(metrics), len(is_ops)     
    # figsize = (2.5*len(is_ops),2*len(metrics))
    if len(metrics) == 2:
        figsize = (5,4)
    elif len(metrics) == 1:
        figsize = (5, 2.3)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=True)  # sharey=False
    axs = matrixify_axs(axs, nrows, ncols)  # convert axs to 2D array
    # label_axs(fig, axs)  # alphabetically label subfigures             

    model_types_plotted = []
    model_types_seeds = {}
    comparison_bleu_methods_by_metric = {
        metric: set() for metric in metrics
    }
    max_plot_epoch = 0
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        ax = axs[row_idx, col_idx] 
        comparison_bleu_methods = comparison_bleu_methods_by_metric[metric]
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
                elif model_type[-8:] == 'dpformer':
                    color = '#636363'
                elif model_type[-10:] == 'sinkformer':
                    color = 'k'
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
                    epochs, run_perf_all = load_seed_runs(
                        model_info['model_dir'].item(), seeds, metric,
                        comparison_bleu_methods=comparison_bleu_methods
                    )
                else:
                    continue

                #EPOCHS_PLOT = 49
                if run_perf_all is not None:
                    metric_curves = get_metric_curves(run_perf_all)
                    plot_epochs = epochs.to_numpy() + 1
                    max_plot_epoch = max(max_plot_epoch, plot_epochs[-1])
                    if model_type[-8:] == 'dpformer':  
                        model_label = 'DP' 
                    elif model_type[-10:] == 'sinkformer':
                        model_label = 'SINK'
                    elif is_fns:
                        model_label = rf'$\alpha = {alpha}$'
                    exe_plot = ax.plot(plot_epochs[start_epoch-1:],
                                       metric_curves[1][start_epoch-1:],
                                       linestyle='-', linewidth=1,
                                       c=color, alpha=1, clip_on=True, 
                                       zorder=1, label=model_label)
                    if (row_idx,col_idx) == (0,0):
                        im = exe_plot                      
                    # Calculate std                       
                    metric_std = metric_curves[2] - metric_curves[1]
                    ax.fill_between(plot_epochs[start_epoch-1:],
                                    metric_curves[0][start_epoch-1:],
                                    metric_curves[2][start_epoch-1:],
                                    color=color, alpha=0.3, clip_on=True, edgecolor='none') 

                    # results of the final epoch
                    row_stats.append([model_type, alpha] +\
                                     final_epoch_stats(run_perf_all))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    #ax.set_xticks([0] + list(range(25,126,25)))
                if not is_fns:
                    break  # only do once if model is not FNS type

        summary_stats = pd.DataFrame(data=row_stats, columns=['model_type','alpha']+stats_colnames)

        # print message
        print(metric)
        print(f'is_op = {is_op}')
        print(summary_stats)
        print('\n')                    

    if max_plot_epoch > 0:
        # axs[0,0].set_xlim([0,max_plot_epoch])
        axs[0,0].set_xlim([start_epoch,max_plot_epoch])

    for cidx in range(ncols):
        axs[0,cidx].margins(0)
        if len(metrics) == 2:
            axs[1,cidx].margins(0)

    # legend
    axs[0,0].legend(loc='best', ncol=2, frameon=False)                     

    for row_idx in range(nrows):
        if metrics[row_idx] == 'bleu':
            # axs[row_idx,0].set_ylim(bottom=0)
            # axs[row_idx,0].set_ylim([18, 37])
            pass
        elif metrics[row_idx][-4:] == 'loss':
            # axs[row_idx,0].set_ylim([2.2, 2.8])
            pass
        if metrics[row_idx] == 'lr':
            axs[row_idx,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # axs[row_idx,0].ticklabel_format(useOffset=True, useMathText=True, axis='y')
            # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        for col_idx, is_op in enumerate(is_ops):  
            ax = axs[row_idx, col_idx]
            if row_idx == 0:                
                ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_ops[col_idx] else r'$\mathbf{W}_{Q,K} \notin O(d)$'
                ax.set_title(ax_title)          
            # axs[row_idx,col_idx].sharey(axs[row_idx, 0])
            axs[-1,col_idx].set_xlabel('Epochs')
    # axs[row_idx,0].set_ylabel(NAMES_DICT[metrics[row_idx]])
    ylabel_dict = {'val_loss': 'Test loss', 'train_loss': 'Training loss', 'bleu': 'Test Bleu score (%)',
                   'lr': 'Learning rate'}
    for row_idx, metric in enumerate(metrics):
        axs[row_idx,0].set_ylabel(ylabel_dict[metric])

    # Subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.115, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

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
    if len(metrics) == 2:
        fig_file = models_root.split('/')[1] + '-' + 'phase_ensembles'
    elif len(metrics) == 1:
        fig_file = models_root.split('/')[1] + '-' + f'{metrics[0]}'
    fig_file += '.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')
    # plt.show()    


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
