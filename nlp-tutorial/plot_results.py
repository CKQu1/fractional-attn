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
from UTILS.mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history
from UTILS.mutils import collect_model_dirs, find_subdirs, load_model_files
from UTILS.figure_utils import matrixify_axs, label_axs

import re
from pathlib import Path

matplotlib.use("Agg")

# ---------- Global plot settings ----------
# font_type = {'family' : 'sans-serif'}
# plt.rc('font', **font_type)
# plt.rc('legend',fontsize=7)
#linestyles = ['solid', 'densely dashed', 'dashed', 'densely dotted', 'dotted']
linestyles = ['-', '--', '-.', ':']
#linestyles = ['-', '--', ':']
markers = ['s', 'D', 'd', 'v', '^', 'o', '.']
markersize = '3'
colors = list(mcl.TABLEAU_COLORS.keys())
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

# Ablation study of Fracformer on sequence length
def fns_seq_len(root, dataset_name='imdb', 
                L=1, hidden=16,
                fns_type='oprdfns'+MODEL_SUFFIX, qk_share=True, metrics='val_acc',
                max_lens='256,512,1024,2048,4096',
                include_others=False, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, models_roots, models_root, root_subdirs
    global model_combo, model_combos
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, dataset, df_model
    global model_types, epochs, ensembles
    global other_final_epoch_metrics, fns_final_epoch_metrics    
    global df_stats, summary_stats     

    metrics = str2ls(metrics) 
    max_lens = [int(max_len) for max_len in str2ls(max_lens)]
    display = str2bool(display) 
    include_others = str2bool(include_others)   
    assert len(metrics) == 1

    models_roots = []
    true_max_lens = []
    root_subdirs = os.listdir(root)
    for ii, max_len in enumerate(max_lens):   
        max_len_str = 'None' if max_len == MAX_LENS_DICT[dataset_name] else max_len
        models_root = f'{L}L-hidden={hidden}-max_len={max_len_str}-glove'
        config_proj = 'qqv' if qk_share else 'qkv'
        if models_root in root_subdirs:
            models_roots.append(njoin(root,models_root,f'config_{config_proj}',dataset_name,f'layers={L}-heads=1-{config_proj}'))
            true_max_lens.append(max_len)
    max_lens = true_max_lens

    print(models_roots)

    suffix = fns_type.split('fns')[-1]
    DCT_ALL = collect_model_dirs(models_roots[0], suffix=suffix)
    model_types = list(DCT_ALL.keys())
    for model_type in model_types:
        if 'fns' in model_type:
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            df_model.reset_index(drop=True, inplace=True)
            break

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = df_model.loc[0,['n_heads', 'n_layers', 'hidden']]
    dataset = df_model.loc[0,'dataset_name']

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small     
    epss = sorted(df_model.loc[:,'bandwidth'].unique())

    #nrows, ncols = len(models_roots), 1
    nrows, ncols = 1, 1

    #figsize = (3*ncols,3*nrows)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'
    
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)     
    
    total_figs = 0
    model_types_plotted = []
    df_stats_data = []
    for row_idx, models_root in enumerate(models_roots):
        inter_seeds = list(set.intersection(*map(set,list(DCT_ALL[fns_type].loc[:,'seeds']))))
        fns_final_epoch_metrics = np.zeros([len(inter_seeds), 2, len(epss), len(alphas)])
        fns_final_epoch_metrics[:] = np.nan
        other_final_epoch_metrics = {}        
        other_best_epoch_metrics = {}

        DCT_ALL = collect_model_dirs(models_root)
        if DCT_ALL == {}:
            continue
        for eps_idx, eps in tqdm(enumerate(epss)):            

            model_type = fns_type
            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            model_df = DCT_ALL[model_type].dropna(subset='alpha')
            model_df.reset_index(drop=True, inplace=True)
            lstyle_model = LINESTYLE_DICT[model_type]

            # -------------------- FNS --------------------            
            row_stats = []
            for alp_idx, alpha in enumerate(alphas):
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & (model_df['ensembles']>0)]
                if len(model_info.index) == 0:
                    continue

                seeds = model_info['seeds'].item()
                qk_share = model_info['qk_share'].item()
                
                #for seed_idx, seed in enumerate(seeds):
                for seed_idx, seed in enumerate(inter_seeds):
                    model_seed_path = njoin(model_info['model_dir'].item(), f'model={seed}')
                    if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                    if isfile(njoin(model_seed_path, '_run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_seed_path, '_run_performance.csv'))

                    # if 'acc' in metrics[0]:
                    #     run_perf.loc[:,metrics[0]] *= 100                     
                    epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int))                

                    fns_final_epoch_metrics[seed_idx, 0, eps_idx, alp_idx] = run_perf.loc[run_perf.index[-1],metrics[0]]                                            
                    if 'acc' in metrics[0]:
                        fns_final_epoch_metrics[seed_idx, 1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].max()  
                    else:
                        fns_final_epoch_metrics[seed_idx, 1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].min()                    

                # print results
                median_metric = np.median(fns_final_epoch_metrics[:,0,eps_idx,alp_idx],0) 
                mean_metric = np.mean(fns_final_epoch_metrics[:,0,eps_idx,alp_idx],0)
                std_metric = np.std(fns_final_epoch_metrics[:,0,eps_idx,alp_idx],0)
                min_metric, max_metric = fns_final_epoch_metrics[:,0,eps_idx,alp_idx].min(), fns_final_epoch_metrics[:,0,eps_idx,alp_idx].max()
                mid_metric = (min_metric + max_metric)/2   
                diff_metric = max_metric - mid_metric                         
                row_stats.append([alpha, eps, min_metric, max_metric, 
                mid_metric, diff_metric, median_metric, mean_metric, std_metric])

            summary_stats = pd.DataFrame(data=row_stats, 
                                         columns=['alpha', 'eps', 'min', 'max', 'mid', 'diff', 'median', 'mean', 'std'])
            print('\n')    
            print(summary_stats)
            #print(fns_final_epoch_metrics)
            print('\n')

        alpha_index = summary_stats.loc[1:,'mean'].argmax() + 1
        df_stats_data.append([summary_stats.loc[alpha_index,'alpha'], summary_stats.loc[alpha_index,'mean'], 
        summary_stats.loc[summary_stats['alpha']==2,'mean'].item()]
        )

        """
        if eps_idx == 0:                
        # -------------------- SINK, DP --------------------                
            other_model_types = ['sink' + suffix, 'dp' + suffix]
            for model_type in other_model_types:
                if model_type in model_types:
                    model_df = DCT_ALL[model_type]
                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = model_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    seeds = model_info['seeds']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:
                        model_seed_path = njoin(model_info['model_dir'], f'model={seeds[0]}')
                        if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                        if isfile(njoin(model_seed_path, '_run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_seed_path, '_run_performance.csv'))

                        if 'acc' in metrics[0]:
                            run_perf.loc[:,metrics[0]] *= 100      
                        epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int)) 

                        other_final_epoch_metrics[model_type] = run_perf.loc[run_perf.index[-1],metrics[0]]
                        if 'acc' in metrics[0]:
                            other_best_epoch_metrics[model_type] = run_perf.loc[:,metrics[0]].max()  
                        else:
                            other_best_epoch_metrics[model_type] = run_perf.loc[:,metrics[0]].min()

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)
        """

    df_stats = pd.DataFrame(data=df_stats_data,
                            columns=['alpha','best_metric','gaussian_metric'])
    if 'acc' in metrics[0]:
        for col_name in df_stats.columns[1:]:
            df_stats.loc[:,col_name] = 100 * df_stats.loc[:,col_name]                            
    print(df_stats)

    ax = axs[0, 0]
    for eps_idx, eps in tqdm(enumerate(epss)):
        # best mean        
        for ii, max_len in enumerate(max_lens):
            ax.plot(max_lens[ii], df_stats.loc[ii,'best_metric'], c=HYP_CMAP(HYP_CNORM(df_stats.loc[ii,'alpha'])),
                    marker='o', markersize=MARKERSIZE-1)

            alpha = df_stats.loc[ii,'alpha']
            ax.annotate(rf'$\alpha$ = {alpha}', (max_lens[ii], df_stats.loc[ii,'best_metric']), size=5.5)

        ax.plot(max_lens, df_stats.loc[:,'best_metric'], c='gray', linestyle='--')

        # gaussian mean                    
        ax.plot(max_lens, df_stats.loc[:,'gaussian_metric'], c=HYP_CMAP(HYP_CNORM(2)),
                marker='s', markersize=MARKERSIZE-1)        

        ax.plot(max_lens, df_stats.loc[:,'gaussian_metric'], c='gray', linestyle='--')

        # ax.fill_between(max_lens, 
        #                 np.quantile(fns_final_epoch_metrics[:,col_idx,eps_idx,:],0,axis=0), 
        #                 np.quantile(fns_final_epoch_metrics[:,col_idx,eps_idx,:],1,axis=0),
        #                 alpha=1/2)   

    if include_others:
        for model_type in other_final_epoch_metrics.keys():
            if col_idx == 0:
                ax.axhline(y=other_final_epoch_metrics[model_type], 
                            linestyle=LINESTYLE_DICT[model_type], c=OTHER_COLORS_DICT[model_type])
            else:
                ax.axhline(y=other_best_epoch_metrics[model_type], 
                            linestyle=LINESTYLE_DICT[model_type], c=OTHER_COLORS_DICT[model_type])

    ax.set_title(NAMES_DICT[metrics[0]])
    ax.set_xlabel('Max Sequence Length')

    # row labels (Q = K)
    title = r'$Q \neq K$' if not qk_share else r'$Q = K$'               
    ax.text(1.2, 0.5, title, transform=(
            ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            va='center', rotation='vertical')  # fontsize='medium',                            

    # subplot labels
    # ax.text(
    #     0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
    #         ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
    #     va='bottom')  # fontsize='medium', fontfamily='sans-serif'

    ax.grid()
    ax.set_xscale('log')
    
    # for model_type in model_types_plotted:   
    #     if 'fns' in model_type:
    #         color = 'k'
    #     elif 'sink' in model_type:
    #         color = OTHER_COLORS[0]
    #     elif 'dp' in model_type:
    #         color = OTHER_COLORS[1]
    #     axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], 
    #                 label=NAMES_DICT[model_type])                  

    ncol_legend = 2 if len(epss) > 1 else 1
    axs[0,0].legend(bbox_to_anchor=(.85, 1.4),   # bbox_to_anchor=(0.85, 1.4)
                    loc='best', ncol=ncol_legend, frameon=False)  
    axs[0,0].xaxis.set_minor_locator(plt.FixedLocator([]))
    axs[0,0].set_xticks(max_lens)
    axs[0,0].set_xticklabels([rf'$2^{{{int(np.log(max_len)/np.log(2))}}}$' for max_len in max_lens])

    axs[0,0].set_ylim([62.55, 63])

    # axs[0,0].legend(loc='upper left', ncol=1, frameon=False)  

    # Add shared x and y labels   
    #fig.supxlabel(r'$\alpha$', fontsize='medium')
    #fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = f'fns_seq_len-L={num_hidden_layers}-H={num_attention_heads}-d={hidden_size}-'            
        fig_file += f'ds={dataset_name_short}'
        # if isfile(njoin(SAVE_DIR, fig_file)):
        #     version = len([fname for fname in os.listdir(SAVE_DIR) if fname==fig_file])
        #     fig_file += f'-v{version}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')


# Ablation study on bandwidth
def fns_fix_eps(models_roots, fns_type='oprdfns'+MODEL_SUFFIX, metrics='val_acc',
                include_others=True, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, dataset, df_model
    global model_types, other_model_types, epochs, ensembles
    global other_final_epoch_metrics, fns_final_epoch_metrics
    global model_types_plotted

    models_roots = str2ls(models_roots)
    model_root_dirs = models_roots
    print(model_root_dirs)

    metrics = str2ls(metrics)    
    display = str2bool(display) 
    include_others = str2bool(include_others)   
    assert len(metrics) == 1

    suffix = fns_type.split('fns')[-1]
    DCT_ALL = collect_model_dirs(model_root_dirs[0], suffix=suffix)
    model_types = list(DCT_ALL.keys())
    for model_type in model_types:
        if 'fns' in model_type:
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            df_model.reset_index(drop=True, inplace=True)
            break

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = df_model.loc[0,['n_heads', 'n_layers', 'hidden']]
    dataset = df_model.loc[0,'dataset_name']    

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small     
    epss = sorted(df_model.loc[:,'bandwidth'].unique())

    nrows, ncols = len(models_roots), 2
    figsize = (3*ncols,3.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'
    
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)     
    
    total_figs = 0
    model_types_plotted = []
    for row_idx, models_root in enumerate(models_roots):
        inter_seeds = list(set.intersection(*map(set,list(DCT_ALL[fns_type].loc[:,'seeds']))))
        fns_final_epoch_metrics = np.zeros([len(inter_seeds), 2, len(epss), len(alphas)])
        fns_final_epoch_metrics[:] = np.nan
        other_final_epoch_metrics = {}        
        other_best_epoch_metrics = {}

        DCT_ALL = collect_model_dirs(models_root)
        if DCT_ALL == {}:
            continue
        for eps_idx, eps in tqdm(enumerate(epss)):            
            model_type = fns_type
            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            model_df = DCT_ALL[model_type].dropna(subset='alpha')
            model_df.reset_index(drop=True, inplace=True)
            lstyle_model = LINESTYLE_DICT[model_type]

            # -------------------- FNS --------------------            
            row_stats = []            
            for alp_idx, alpha in enumerate(alphas):
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & (model_df['ensembles']>0)]
                if len(model_info.index) == 0:
                    continue

                seeds = model_info['seeds'].item()
                qk_share = model_info['qk_share'].item()
                
                #for seed_idx, seed in enumerate(seeds):
                counter = 0
                for seed_idx, seed in enumerate(inter_seeds):
                    model_seed_path = njoin(model_info['model_dir'].item(), f'model={seed}')
                    #print(model_seed_path)  # delete
                    if isfile(njoin(model_seed_path, '_run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_seed_path, '_run_performance.csv'))
                        counter += 1
                    if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                        counter += 1                        

                    if 'acc' in metrics[0]:
                        run_perf.loc[:,metrics[0]] *= 100                     
                    epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int))                

                    fns_final_epoch_metrics[seed_idx, 0, eps_idx, alp_idx] = run_perf.loc[run_perf.index[-1],metrics[0]]                                            
                    if 'acc' in metrics[0]:
                        fns_final_epoch_metrics[seed_idx, 1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].max()  
                    else:
                        fns_final_epoch_metrics[seed_idx, 1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].min()                    

                # print results
                median_metric = np.median(fns_final_epoch_metrics[:,0,eps_idx,alp_idx],0) 
                mean_metric = np.mean(fns_final_epoch_metrics[:,0,eps_idx,alp_idx],0)
                std_metric = np.std(fns_final_epoch_metrics[:,0,eps_idx,alp_idx],0)
                min_metric, max_metric = fns_final_epoch_metrics[:,0,eps_idx,alp_idx].min(), fns_final_epoch_metrics[:,0,eps_idx,alp_idx].max()
                mid_metric = (min_metric + max_metric)/2   
                diff_metric = max_metric - mid_metric                         
                row_stats.append([alpha, eps, min_metric, max_metric, 
                mid_metric, diff_metric, median_metric, mean_metric, std_metric, counter])

            summary_stats = pd.DataFrame(data=row_stats, 
                                         columns=['alpha', 'eps', 'min', 'max', 'mid',\
                                                  'diff', 'median', 'mean', 'std', 'counter'])
            print('\n')    
            print(summary_stats)
            print('\n')

            if eps_idx == 0:                
            # -------------------- SINK, DP --------------------                
                #other_model_types = [model_type for model_type in list(DCT_ALL.keys()) if model_type != fns_type]
                other_model_types = [model_type for model_type in list(DCT_ALL.keys()) if 'dp' in model_type]
                for model_type in other_model_types:

                    model_df = DCT_ALL[model_type]
                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = model_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    seeds = model_info['seeds']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:
                        model_seed_path = njoin(model_info['model_dir'], f'model={seeds[0]}')
                        if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                        if isfile(njoin(model_seed_path, '_run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_seed_path, '_run_performance.csv'))

                        if 'acc' in metrics[0]:
                            run_perf.loc[:,metrics[0]] *= 100      
                        epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int)) 

                        other_final_epoch_metrics[model_type] = run_perf.loc[run_perf.index[-1],metrics[0]]
                        if 'acc' in metrics[0]:
                            other_best_epoch_metrics[model_type] = run_perf.loc[:,metrics[0]].max()  
                        else:
                            other_best_epoch_metrics[model_type] = run_perf.loc[:,metrics[0]].min()

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)            

        for col_idx in range(fns_final_epoch_metrics.shape[1]):
            ax = axs[row_idx, col_idx]
            for eps_idx, eps in tqdm(enumerate(epss)):
                # median
                # median_metric = np.median(fns_final_epoch_metrics[:,col_idx,eps_idx,:],0)    
                # ax.plot(alphas, median_metric, label = rf'$\varepsilon$ = {eps}',
                #         linestyle=f'--', marker=markers[eps_idx],markersize=MARKERSIZE)

                # ax.fill_between(alphas, 
                #                 np.quantile(fns_final_epoch_metrics[:,col_idx,eps_idx,:],0,axis=0), 
                #                 np.quantile(fns_final_epoch_metrics[:,col_idx,eps_idx,:],1,axis=0),
                #                 alpha=1/2)  

                # mean
                metric_mean = np.mean(fns_final_epoch_metrics[:,col_idx,eps_idx,:],0)
                ax.plot(alphas, metric_mean, 
                        linestyle=LINESTYLE_DICT[model_type], c='blue',
                        marker=markers[eps_idx],markersize=MARKERSIZE)

                metric_std = np.std(fns_final_epoch_metrics[:,col_idx,eps_idx,:],0)
                ax.fill_between(alphas, 
                                metric_mean - metric_std, 
                                metric_mean + metric_std,                                
                                color='blue', alpha=1/2)  

                # best/worst iteration
                # ax.plot(alphas, np.max(fns_final_epoch_metrics[:,col_idx,eps_idx,:],0), label = rf'$\varepsilon$ = {eps}',
                #         linestyle=f'--', marker=markers[eps_idx],markersize=MARKERSIZE)     

            if include_others:
                for model_type in other_final_epoch_metrics.keys():
                    if col_idx == 0:
                        ax.axhline(y=other_final_epoch_metrics[model_type], 
                                   linestyle=LINESTYLE_DICT[model_type], c='red')
                        # c=OTHER_COLORS_DICT[model_type]
                    else:
                        ax.axhline(y=other_best_epoch_metrics[model_type], 
                                   linestyle=LINESTYLE_DICT[model_type], c='red')

            if row_idx == 0:
                title = 'Final' if col_idx==0 else 'Best'
                title += ' ' + NAMES_DICT[metrics[0]]
                ax.set_title(title)
            elif row_idx == nrows - 1:
                ax.set_xlabel(rf'$\alpha$')

            # row labels (Q = K)
            if col_idx == ncols - 1:
                title = r'$Q \neq K$' if not qk_share else r'$Q = K$'               
                ax.text(1.2, 0.5, title, transform=(
                        ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                        va='center', rotation='vertical')  # fontsize='medium',                            

            # subplot labels
            ax.text(
                0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom')  # fontsize='medium', fontfamily='sans-serif'

            ax.grid()  #ax.yaxis.grid(True)            
            ax.set_xticks(alphas)

            total_figs += 1
    
    for model_idx, model_type in enumerate(model_types_plotted):
        if 'fns' in model_type:
            axs[0,0].plot([], [], linestyle='--', c='blue', label=NAMES_DICT[model_type])
        else:
            axs[0,0].plot([], [], linestyle=LINESTYLE_DICT[model_type], 
                          c='r', label=NAMES_DICT[model_type])             

    #ncol_legend = 2 if len(epss) > 1 else 1
    if len(epss) > 1:
        axs[0,0].plot([], [], label = rf'$\varepsilon$ = {eps}')
    axs[0,0].legend(bbox_to_anchor=(0.85, 1.4),   # bbox_to_anchor=(0.85, 1.4)
                    loc='best', ncol=len(model_types_plotted), frameon=False)  #ncol=ncol_legend

    # Add shared x and y labels   
    #fig.supxlabel(r'$\alpha$', fontsize='medium')
    #fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'

        fig_file += f'fix_eps-{fns_type}-'
        if len(model_root_dirs) == 1:
            fig_file += f'qk_share={qk_share}-'
        else:
            fig_file += 'qk_share=both-'
        fig_file += f'layers={num_hidden_layers}-'
        fig_file += f'heads={num_attention_heads}-hidden={hidden_size}-'
        fig_file += f'ds={dataset_name_short}'
        # if isfile(njoin(SAVE_DIR, fig_file)):
        #     version = len([fname for fname in os.listdir(SAVE_DIR) if fname==fig_file])
        #     fig_file += f'-v{version}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')


# Plots average of metrics over ensembles
# assumption 1 and 2 possibilities for full-sized models
"""
python -i plot_results.py plot_ensembles .droot/formers_trained/layers=2-heads=8-hidden=768-epochs=10-qkv/
"""
def phase_ensembles(models_root, selected_dataset='imdb',
                    fns_manifold='rd', qk_share=False, selected_alphas='1,2',
                    metrics='val_acc,val_loss',
                    is_ops = [True],
                    #is_ops = [True],
                    cbar_separate=False, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos, model_types_plotted, model_types_short
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, run_perf_all, dataset, df_model
    global model_types, model_info, epochs, ensembles, seeds
    global models_roots, qk_shares, matching_df
    global model_type, other_model_types
    global DCT_cur, df_model_cur, other_matching_df
    global model_seed_path    
    global summary_stats, row_stats, metric_m

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    qk_share = qk_share if isinstance(qk_share, bool) else literal_eval(qk_share)

    models_roots = []
    subdir_l1s = [ f.path for f in os.scandir(models_root) if f.is_dir() ]
    for subdir_l1 in subdir_l1s:
        subdir_l2s = [ f.path for f in os.scandir(njoin(subdir_l1)) if f.is_dir() and 'config_' in f.path ]
        for subdir_l2 in subdir_l2s:
            subdir_l3s = [ f.path for f in os.scandir(subdir_l2) if f.is_dir() ]
            for subdir_l3 in subdir_l3s:
                models_roots.append(subdir_l3)

    model_root_dirs = models_roots
    print(model_root_dirs)        

    metrics = str2ls(metrics)    
    display = str2bool(display)    

    #suffix = fns_type.split('fns')[-1]
    suffix = MODEL_SUFFIX
    model_types = []   
    DCT_ALL = {} 
    for ii, model_root_dir in enumerate(model_root_dirs):
        DCT_cur = collect_model_dirs(model_root_dir, suffix=suffix)
        #print(DCT_ALL)
        model_types_cur = list(DCT_cur.keys())
    
        for model_type in model_types_cur:
            df_model_cur = DCT_cur[model_type]
            #if 'fns' in model_type:
            if model_type not in model_types:
                model_types.append(model_type)
                if 'alpha' in df_model_cur.columns:
                    DCT_ALL[model_type] = df_model_cur.dropna(subset='alpha')
                else:
                    DCT_ALL[model_type] = df_model_cur
            else:                
                if 'alpha' in df_model_cur.columns:
                    DCT_ALL[model_type] = DCT_ALL[model_type].append(df_model_cur.dropna(subset='alpha'), ignore_index=True)
                else:
                    DCT_ALL[model_type] = DCT_ALL[model_type].append(df_model_cur, ignore_index=True)

    df_model = DCT_ALL[[model_type for model_type in list(DCT_ALL.keys()) if fns_manifold in model_type][0]]
    df_model.reset_index(drop=True, inplace=True)
    
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
    eps = epss[0]

    #metric = metrics[0]
    qk_shares = list(df_model.loc[:,'qk_share'].unique())    
    if isinstance(is_ops, str):
        is_ops = str2ls(is_ops)
    nrows, ncols = len(metrics), len(is_ops)     
    figsize = (3*ncols,3.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=False)  # layout='constrained'    
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)                 

    model_types_plotted = []
    model_types_seeds = {}     
    for row_idx, metric in enumerate(metrics):
        for col_idx, is_op in enumerate(is_ops):

            # summary statistics
            row_stats = []

            ax = axs[row_idx, col_idx]
            model_type = 'op' + fns_manifold + 'fns' + MODEL_SUFFIX if is_op else fns_manifold + 'fns' + MODEL_SUFFIX
            print(f'model_type = {model_type}')
            df_model = DCT_ALL[model_type]
            matching_df = df_model[(df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&
                                    (df_model['is_op']==is_op)&
                                    (df_model['model_dir'].str.contains(f'/{model_type}-'))
                                    #(df_model['model_dir']==model_type)
                                    ]
            #print(matching_df)
            #quit()  # delete

            if matching_df.shape[0] > 0: 
                DCT_matching = { model_type: matching_df }
            else:
                DCT_matching = {}

            if DCT_matching == {}:
                continue

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            for alpha in alphas:                
                model_df = DCT_matching[model_type].dropna(subset='alpha')
                model_df.reset_index(drop=True, inplace=True)
                lstyle_model = LINESTYLE_DICT[model_type]

                # -------------------- FNS --------------------                        
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & 
                                      (model_df['ensembles']>0) & model_df['model_dir'].str.contains(selected_dataset)]
                if len(model_info.index) == 0:
                    continue

                seeds = model_info['seeds'].item()
                qk_share = model_info['qk_share'].item()
                
                run_perf_all = []
                counter = 0
                for seed in seeds:
                    model_seed_path = njoin(model_info['model_dir'].item(), f'model={seed}')
                    if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                    if isfile(njoin(model_seed_path, '_run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_seed_path, '_run_performance.csv'))            

                    epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int))                
                    if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                        run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100
                        counter += 1

                    run_perf_all.append(run_perf.loc[:,metric])

                run_perf_all = pd.concat(run_perf_all, axis=1)

                metric_m = run_perf_all.quantile(0.5,1)
                metric_l = run_perf_all.quantile(0.25,1)
                metric_u = run_perf_all.quantile(0.75,1)

                trans = 1
                #trans = HYP_TRANS(alpha)
                if alpha in selected_alphas:  # only plot selected_alphas
                    if (row_idx,col_idx) == (0,0):
                        im = ax.plot(epochs, metric_m, linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                                
                        ax.fill_between(epochs, metric_l, metric_u,
                                        color=c_hyp, alpha=trans/2)
                    else:
                        ax.plot(epochs, metric_m, linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                            
                        ax.fill_between(epochs, metric_l, metric_u,
                                        color=c_hyp, alpha=trans/2)                                                                        

                # results of the final epoch
                epoch_index = run_perf_all.index[-1]
                metric_min = run_perf_all.loc[epoch_index,metric].min()
                metric_max = run_perf_all.loc[epoch_index,metric].max()
                metric_mid = (metric_min + metric_max) / 2
                if run_perf_all.shape[0] == 1:
                    metric_median = run_perf_all.loc[epoch_index,metric].median()
                    metric_mean = run_perf_all.loc[epoch_index,metric].mean()
                    metric_std = run_perf_all.loc[epoch_index,metric].std()
                else:
                    metric_median = run_perf_all.loc[epoch_index,metric]
                    metric_mean = run_perf_all.loc[epoch_index,metric]
                    metric_std = 0
                row_stats.append([alpha, metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std, counter])

            summary_stats = pd.DataFrame(data=row_stats, 
            columns=['alpha', 'min', 'max', 'mid', 'median', 'mean', 'std', 'counter']
            )

            print(metric)
            print(f'is_op = {is_op}, qk_share = {qk_share}')
            print(summary_stats)
            print('\n')

            # if row_idx != nrows - 1:
            #     ax.set_xticklabels([])

            # col labels (bandwidth)
            # if row_idx == 0:
            #     ax.set_title(rf'$\varepsilon = {{{eps}}}$')
            # row labels (Q = K)
            # if col_idx == ncols - 1:
            #     title = r'$Q \neq K$' if not qk_share else r'$Q = K$'               
            #     ax.text(1.2, 0.5, title, transform=(
            #                     ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            #                     va='center', rotation='vertical')  # fontsize='medium',                          

            # -------------------- SINK, DP --------------------                
            #other_model_types = ['sink' + suffix, 'dp' + suffix]
            if is_op:
                #other_model_types = ['opdp' + suffix, 'opsink' + suffix]
                other_model_types = ['opdp' + suffix]
                #other_model_types = ['opsink' + suffix]
            else:
                #other_model_types = ['dp' + suffix, 'sink' + suffix]
                other_model_types = ['dp' + suffix]
                #other_model_types = ['sink' + suffix]
            for oidx, model_type in enumerate(other_model_types):
                # summary statistics
                row_stats = []

                if model_type in model_types:
                    model_df = DCT_ALL[model_type]
                    other_matching_df = model_df[(model_df['ensembles']>0)&(model_df['qk_share']==qk_share)&
                                                 (model_df['is_op']==is_op)&
                                                 (model_df['model_dir'].str.contains(model_type))]                

                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = other_matching_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    seeds = model_info['seeds']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:

                        run_perf_all = []
                        counter = 0
                        for seed in seeds:
                            model_seed_path = njoin(model_info['model_dir'], f'model={seed}')
                            if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                                counter += 1
                            # if isfile(njoin(model_seed_path, '_run_performance.csv')): 
                            #     run_perf = pd.read_csv(njoin(model_seed_path, '_run_performance.csv'))

                            epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int)) 
                            if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                                run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100
                        
                            run_perf_all.append(run_perf.loc[:,metric])
                        run_perf_all = pd.concat(run_perf_all, axis=1)

                        metric_m = run_perf_all.quantile(0.5,1)
                        metric_l = run_perf_all.quantile(0,1)
                        metric_u = run_perf_all.quantile(1,1)
                        trans = 1

                        # print(model_type + f' qk_share = {qk_share}, is_op = {is_op}')
                        # print(metric_m)

                        ax.plot(epochs, metric_m, linestyle=lstyle_model, 
                                c=OTHER_COLORS_DICT[model_type], alpha=trans)    
                        ax.plot(epochs, metric_l, linestyle='-', linewidth=0.8,
                                c=OTHER_COLORS_DICT[model_type], alpha=trans)  
                        ax.plot(epochs, metric_u, linestyle='-', linewidth=0.8,
                                c=OTHER_COLORS_DICT[model_type], alpha=trans)                                                                  
                                            
                        ax.fill_between(epochs, metric_l, metric_u,
                                        color=OTHER_COLORS_DICT[model_type], alpha=trans/2)    

                    # results of the final epoch
                    epoch_index = run_perf_all.index[-1]
                    metric_min = run_perf_all.loc[epoch_index,metric].min()
                    metric_max = run_perf_all.loc[epoch_index,metric].max()
                    metric_mid = (metric_min + metric_max) / 2
                    metric_median = run_perf_all.loc[epoch_index,metric].median()
                    metric_mean = run_perf_all.loc[epoch_index,metric].mean()
                    metric_std = run_perf_all.loc[epoch_index,metric].std()
                    row_stats.append([metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std, counter])

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

                summary_stats = pd.DataFrame(data=row_stats, 
                columns=['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']
                )

                print(model_type)
                print(metric)
                print(f'is_op = {is_op}, qk_share = {qk_share}')
                print(summary_stats)
                print('\n')

            ax.grid()
            #ax.axvline(x=15, color='k', linestyle='--', linewidth=0.8)
            #ax.yaxis.grid(True)        

    # axs[0,0].set_ylim([75,85])
    # axs[1,0].set_ylim([0.45,0.65])
    # axs[1,0].set_xticks([5,10,15,20])

    # labels
    model_labels = []
    for model_type in model_types_plotted:   
        color = 'k' if 'fns' in model_type else OTHER_COLORS_DICT[model_type]            
        model_label = NAMES_DICT[model_type]
        if model_label not in model_labels:            
            axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], 
                          label=model_label)

            model_labels.append(model_label)

    # legend
    if selected_alphas is None:
        for alpha in alphas[::-1]:
            axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
                        label=rf'$\alpha$ = {alpha}')   
    else:
        for alpha in selected_alphas[::-1]:
            axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
                        label=rf'$\alpha$ = {alpha}')           

    #ncol_legend = 2 if len(model_types_plotted) == 3 else 1
    ncol_legend = 2
    if len(model_types_plotted) >= 2:
        #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
        axs[0,0].legend(bbox_to_anchor=(0.95, 1.35),   # bbox_to_anchor=(0.85, 1.35)
                        loc='best', ncol=ncol_legend, frameon=False)                     

    # Add shared x and y labels     
    total_figs = 0      
    #fig.supxlabel('Epochs')  # , fontsize='medium'
    #fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    for row_idx in range(len(qk_shares)):        
        for col_idx, is_op in enumerate(is_ops):  
            ax = axs[row_idx, col_idx]
            #ax.set_ylabel(NAMES_DICT[metric])
            if row_idx == 0:
                #ax.set_title(NAMES_DICT[metric])
                ax_title = r'$W \in O(d)$' if is_ops[col_idx] else r'$W \notin O(d)$'
                ax.set_title(ax_title)

            # subplot labels
            ax.text(
                0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom')  # , fontfamily='sans-serif', fontsize='medium',     

            total_figs += 1
            
            axs[row_idx,col_idx].sharey(axs[row_idx, 0])
            axs[-1,col_idx].set_xlabel('Epochs')
        axs[row_idx,0].set_ylabel(NAMES_DICT[metrics[row_idx]])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

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
        fig_file += f'l={num_hidden_layers}-d={hidden_size}-qk_share={qk_share}-'
        fig_file += '_'.join(model_types_short)+ '-' + metrics[0] + '-' + f'ds={dataset_name_short}'
        # if isfile(njoin(SAVE_DIR, fig_file)):
        #     version = len([fname for fname in os.listdir(SAVE_DIR) if fname==fig_file])
        #     fig_file += f'-v{version}'
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


def hyperparam_effects(models_root, fns_manifold='rd', is_rescale_dist=True,
                       qk_shares=[False, True], selected_alphas='1.2,2',
                       metric='val_acc', selected_dataset='imdb',
                       is_op=True):

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
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'    
    axs = matrixify_axs(axs, nrows, ncols)            

    # (model_types, qk_share, L, d_model)
    N_model_types = len(selected_alphas) + 1
    average_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    std_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    average_metric_matrix[:] = np.nan
    std_metric_matrix[:] = np.nan
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
        for _ in range(2):
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
                    #counter = run_perf_all.shape[1]
                    metric_curves = get_metric_curves(run_perf_all)  

                if run_perf_all is not None:
                    average_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmean(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                        #np.nanmedian(run_perf_all.loc[run_perf_all.index[-1]:,metric])                                                
                    std_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanstd(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                    counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        (~np.isnan(run_perf_all.loc[run_perf_all.index[-1]:,metric].to_numpy())).sum()                        
                    nan_counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        (np.isnan(run_perf_all.loc[run_perf_all.index[-1]:,metric].to_numpy())).sum()

                if not is_fns:
                    break  # only do once if model is not FNS type                            

    # average_metric_matrix[average_metric_matrix==0] = np.nan
    # counter_matrix[counter_matrix==0] = np.nan

    axs[0,0].set_xscale('log')
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row,col]
            for aidx in range(len(selected_alphas) + 1):
                average_metrics = average_metric_matrix[row,aidx,col]
                std_metrics = std_metric_matrix[row,aidx,col]
                mask = ~np.isnan(average_metrics)

                is_fns = aidx < len(selected_alphas)
                # color                                 
                if is_fns:
                    alpha = selected_alphas[aidx]
                    legend_label = rf'$\alpha={selected_alphas[aidx]}$'
                    color = HYP_CMAP(HYP_CNORM(alpha))  
                    model_type = fns_type                  
                else:
                    legend_label = 'Transformer'
                    model_type = other_model_type
                    color = OTHER_COLORS_DICT[model_type]
                linestyle = LINESTYLE_DICT[model_type]

                ax.plot(emb_ds[mask], average_metrics[mask], 
                        c=color, linestyle=linestyle, label=legend_label)
                                
                # ax.fill_between(emb_ds[mask], 
                #                 average_metrics[mask] - std_metrics[mask], 
                #                 average_metrics[mask] + std_metrics[mask],
                #                 color=color, alpha=1/2) # linestyle=linestyle, 

                if (row,col,aidx) == (0,0,0):
                    axs[0,0].plot([],[],
                                  linestyle=LINESTYLE_DICT[model_type],c='k',
                                  label=NAMES_DICT[model_type])

                ax.set_xticks(list(emb_ds))
                ax.set_xticklabels(list(emb_ds))    
                ax.xaxis.set_minor_formatter(NullFormatter())                          

    axs[0,0].legend(frameon=False,ncols=2)
    #axs[0,0].set_ylim([70,85.5])

    for col in range(ncols):
        #axs[0,col].set_title(rf'{NAMES_DICT[fns_type]} ($\alpha = {selected_alphas[col]}$)')
        axs[0,col].set_title(rf'$L = {layers[col]}$')
    # if other_model_type in DCT_ALL.keys():        
    #     axs[0,ncols].set_title(f'{NAMES_DICT[other_model_type]}')
    for row in range(nrows):
        qk_share = qk_shares[row]
        if qk_share:
            axs[row,0].set_ylabel(r'$Q = K$')
        else:
            axs[row,0].set_ylabel(r'$Q \neq K$')

    # alphabetically label subfigures          
    label_axs(fig, axs)
    # x/y axis label
    #fig.supxlabel('Embedding Dimension'); fig.supylabel('Layers')
    # for row in range(axs.shape[0]):
    #     axs[row,0].set_ylabel(r'Layer $L$')
    for col in range(axs.shape[1]):
        axs[-1,col].set_xlabel(r'Dimension $d$')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    # dataset_name_short = ''
    # if isinstance(dataset,str):
    #     if '_' in dataset:
    #         for s in dataset.split('_'):
    #             dataset_name_short += s[0]
    #     else:
    #         dataset_name_short += dataset

    # model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')    

    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
    #fig_file = models_root.split('/')[1] + '-'           
    fig_file = 'hyperparam_effects'
    # fig_file += f'l=' + '+'.join(layers) + '-'
    # fig_file += f'd=' + '+'.join(emb_ds) + '-'
    # fig_file += f'qk_share' + '+'.join(qk_shares)
    #fig_file += '_'.join(model_types_short)+ '-' + metrics[0]'
    fig_file += '.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file))            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')


# for plotting dynamic inference
def dynamic_inference(models_root, n_layer=1,
                      fns_type='fns', manifold='rd', is_rescale_dist=True, selected_alphas=[1.2, 2.0],
                      is_op=True, qk_shares=[False,True], metric='test_acc',
                      batch_size=64, is_dist_based=False):

    global model_dirs, emb_ds

    # general setting
    batch_size = int(batch_size)
    is_dist_based = str2bool(is_dist_based)    
    fname = 'dist' if is_dist_based else 'prob'
    fname += f'-bs={batch_size}-inference.csv'

    # get layers, emb_ds
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
    controlled_vars = inference.loc[:,'controlled_variable']
    N_control_var = len(controlled_vars)
    ensembles = 5  # figure out how to extract this

    nrows, ncols = len(qk_shares), len(emb_ds)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'
    axs = matrixify_axs(axs, nrows, ncols)
    label_axs(fig, axs)

    metrics_dynamic = np.zeros([len(selected_alphas)+1, len(qk_shares), 
                                len(emb_ds), N_control_var, ensembles])
    metrics_dynamic[:] = np.nan
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
        inference = pd.read_csv(njoin(model_dir, fname))
        metrics_dynamic[alpha_idx, qk_shares.index(qk_share), list(emb_ds).index(hidden), :, seed] =\
              inference.loc[:,metric]

    for sidx, didx, alpha_idx in\
          product(range(len(qk_shares)), range(len(emb_ds)), range(len(selected_alphas)+1)):
        is_fns = alpha_idx < len(selected_alphas)
        if is_fns:
            alpha = selected_alphas[alpha_idx]
            color = HYP_CMAP(HYP_CNORM(alpha))
        else:
            color = OTHER_COLORS_DICT[other_type]

        metric_mean = np.nanmean(metrics_dynamic[alpha_idx,sidx,didx,:,:],-1)
        metric_std = np.nanstd(metrics_dynamic[alpha_idx,sidx,didx,:,:],-1)
                            
        axs[sidx,didx].plot(controlled_vars, metric_mean,
                            markersize=MARKERSIZE,
                            c=color, linestyle=LINESTYLE_DICT[fns_type])  

        # axs[sidx,didx].fill_between(controlled_vars,  metric_mean - metric_std, metric_mean + metric_std,
        #                             color=color, alpha=1/2)                           

    # legends
    for alpha_idx, alpha in enumerate(selected_alphas):
        c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
        axs[0,0].plot([], [], c=c_hyp, linestyle=LINESTYLE_DICT[fns_type],
                    label=rf'$\alpha$ = {alpha}')    
    
    axs[0,0].plot([], [], c=OTHER_COLORS_DICT[other_type],
                  linestyle=LINESTYLE_DICT[other_type])                      
                        
    control_var_name = 'Distance threshold' if is_dist_based else 'Removal probability'
    for ncol in range(ncols):
        axs[0,ncol].set_title(rf'$d = {emb_ds[ncol]}$')
        axs[-1,ncol].set_xlabel(control_var_name)
    for nrow in range(nrows):
        axs[nrow,0].set_ylabel(r'$Q = K$' if qk_shares[nrow] else r'$Q \neq K$')
    axs[0,0].legend(frameon=False)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label   

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
        fig_file += f'dynamic_inference_dist.pdf'
    else:
        fig_file += f'dynamic_inference_prob.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file))            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')        

# ---------------------------------------- END -----------------------------------------------------

def hyperparam_phase(models_root, fns_manifold='rd', is_rescale_dist=False,
                     qk_shares=[False, True], selected_alphas='1,2',
                     metric='val_acc',
                     is_op=True):

    global layer_dirs_dict, DCT_ALL, setting_dir
    global metric_matrix, counter_matrix, average_metric_matrix, run_perf, other_matching_df
    global layers, emb_ds
    global other_model_type, other_model_condition, other_model_info, other_matching_df, other_model_df
    global model_df, run_perf
    global qk_share

    assert fns_manifold in ['sphere', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    assert metric in ['train_acc', 'train_loss', 'val_acc', 'val_loss']    
    fns_type = fns_manifold + 'fns' + MODEL_SUFFIX 
    other_model_type = 'dpformer'
    if is_op:
        fns_type = 'op' + fns_type
        other_model_type = 'op' + other_model_type
    is_op = str2bool(is_op)
    is_rescale_dist = str2bool(is_rescale_dist)
    qk_shares = str2ls(qk_shares)        
    #display = str2bool(display)  
    selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
    eps = 1

    # Regular expression pattern
    pattern = r"\d+L-hidden=\d+-max_len=None"
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
            emb_d = int(layer_dir.split('-')[1].split('=')[1])  
            if isdir(njoin(models_root, layer_dir)):
                layer_dirs_dict[f'{layer}-{emb_d}'] = njoin(models_root, layer_dir)
            layers.append(layer)
            emb_ds.append(emb_d)
    layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 6]
    emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 256]
    
    nrows, ncols = len(qk_shares), len(selected_alphas) + 1
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'    
    axs = matrixify_axs(axs, nrows, ncols)            

    # (model_types, qk_share, L, d_model)
    metric_matrix = np.zeros([nrows, ncols, len(layers), len(emb_ds)])
    counter_matrix = np.zeros([nrows, ncols, len(layers), len(emb_ds)])
    for qk_ii, qk_share in enumerate(qk_shares):
        qk_share_dirname = 'config_qqv' if qk_share else 'config_qkv'
        for layer_idx, layer in enumerate(layers):
            for emb_d_idx, emb_d in tqdm(enumerate(emb_ds)):
                print(f'qk_share = {qk_share}, layer = {layer}, emb_d = {emb_d}')    
                # directories matching the above setting in the triple for loop
                if f'{layer}-{emb_d}' in layer_dirs_dict.keys():
                    if qk_share_dirname in os.listdir(layer_dirs_dict[f'{layer}-{emb_d}']):
                        setting_dir = njoin(layer_dirs_dict[f'{layer}-{emb_d}'], qk_share_dirname)
                    else:
                        continue
                else:
                    continue
                for _ in range(2):
                    setting_dir = njoin(setting_dir, os.listdir(setting_dir)[0])
                DCT_ALL = collect_model_dirs(setting_dir, suffix=MODEL_SUFFIX)
                model_df = DCT_ALL[fns_type].dropna(subset='alpha')
                model_df.reset_index(drop=True, inplace=True)
                for alpha_idx, alpha in enumerate(selected_alphas):             
                    c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color 
                    model_df = DCT_ALL[fns_type].dropna(subset='alpha')
                    model_df.reset_index(drop=True, inplace=True)
                    lstyle_model = LINESTYLE_DICT[fns_type]

                    # -------------------- FNS --------------------                                         
                    model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & 
                                          (model_df['ensembles']>0)]
                    if len(model_info.index) == 0:
                        continue

                    seeds = model_info['seeds'].item()                    
                    for seed in seeds:
                        model_seed_path = njoin(model_info['model_dir'].item(), f'model={seed}')
                        if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))        
                            if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                                run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100

                            metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] += run_perf.loc[run_perf.index[-1],metric]
                            counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] += 1

                # -------------------- SINK, DP --------------------     
                if other_model_type in DCT_ALL.keys():    
                    other_model_df = DCT_ALL[other_model_type]                        
                    other_model_condition = (other_model_df['ensembles']>0) &\
                                            (other_model_df['qk_share']==qk_share) &\
                                            (other_model_df['is_op']==is_op) &\
                                            (other_model_df['model_dir'].str.contains(f'/{other_model_type}'))
                    #print(f'other_model_condition: {other_model_condition}')
                    other_matching_df = other_model_df[other_model_condition]             

                    if other_matching_df.shape[0] > 0:
                        other_model_info = other_matching_df.iloc[0,:]
                        seeds = other_model_info['seeds']
                        for seed in seeds:
                            model_seed_path = njoin(other_model_info['model_dir'], f'model={seed}')
                            if isfile(njoin(model_seed_path, 'run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_seed_path, 'run_performance.csv'))
                                if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                                    run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100     

                                metric_matrix[qk_ii,len(selected_alphas),layer_idx,emb_d_idx] += run_perf.loc[run_perf.index[-1],metric]
                                counter_matrix[qk_ii,len(selected_alphas),layer_idx,emb_d_idx] += 1                                

    metric_matrix[metric_matrix==0] = np.nan
    counter_matrix[counter_matrix==0] = np.nan
    average_metric_matrix = metric_matrix / counter_matrix

    emb_d_powers = [int(np.log(emb_d)/np.log(2)) for emb_d in emb_ds]

    centers = [emb_d_powers[0], emb_d_powers[-1], layers[0], layers[-1]]
    dx, = np.diff(centers[:2])/(metric_matrix[0,0].shape[1]-1)
    dy, = -np.diff(centers[2:])/(metric_matrix[0,0].shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]

    # Define color normalization for different rows
    cmap = 'jet'
    norms = []
    for row in range(nrows):
        norm = plt.Normalize(vmin=np.nanmin(average_metric_matrix[row]),
                             vmax=np.nanmax(average_metric_matrix[row]))
        norms.append(norm)
        for col in range(ncols):
            axs[row,col].imshow(average_metric_matrix[row, col], cmap=cmap, norm=norm,
                                extent=extent, aspect='auto', origin='upper')  # origin='lower',

    for col in range(ncols-1):
        axs[0,col].set_title(rf'{NAMES_DICT[fns_type]} ($\alpha = {selected_alphas[col]}$)')
    axs[0,ncols-1].set_title(f'{NAMES_DICT[other_model_type]}')

    # Get rightmost column position to align colorbars
    pos_top = axs[0, -1].get_position()
    pos_bottom = axs[1, -1].get_position()
    cbar_x = pos_top.x1 + 0.02  # Shift right of the rightmost column
    cbar_width = 0.02  # Set colorbar width

    # Align first-row colorbar with top row
    cbar_ax1 = fig.add_axes([cbar_x, pos_top.y0, cbar_width, pos_top.y1 - pos_top.y0])  # [left, bottom, width, height]
    fig.colorbar(plt.cm.ScalarMappable(norm=norms[0], cmap=cmap), cax=cbar_ax1, orientation='vertical')

    # Align second-row colorbar with bottom row
    cbar_ax2 = fig.add_axes([cbar_x, pos_bottom.y0, cbar_width, pos_bottom.y1 - pos_bottom.y0])
    fig.colorbar(plt.cm.ScalarMappable(norm=norms[1], cmap=cmap), cax=cbar_ax2, orientation='vertical')

    # x/y ticks and tick labels
    #axs[-1,0].set_xticks(emb_d_powers); axs[-1,0].set_yticks(layers)
    axs[-1,0].set_xticks(np.arange(centers[0], centers[1]+dx, dx))
    axs[-1,0].set_yticks(np.arange(centers[3], centers[2]+dy, dy))

    #axs[-1,0].set_xticklabels([rf'$2^{{{emb_d_power}}}$' for emb_d_power in emb_d_powers])
    axs[-1,0].set_xticklabels(emb_ds)
    axs[-1,0].set_yticklabels(layers)
    # alphabetically label subfigures          
    label_axs(fig, axs)
    # x/y axis label
    #fig.supxlabel('Embedding Dimension'); fig.supylabel('Layers')
    for row in range(axs.shape[0]):
        axs[row,0].set_ylabel(r'Layer $L$')
    for col in range(axs.shape[1]):
        axs[-1,col].set_xlabel(r'Dimension $d$')

    # Adjust layout
    #plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    # dataset_name_short = ''
    # if isinstance(dataset,str):
    #     if '_' in dataset:
    #         for s in dataset.split('_'):
    #             dataset_name_short += s[0]
    #     else:
    #         dataset_name_short += dataset

    # model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')    

    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
    #fig_file = models_root.split('/')[1] + '-'           
    fig_file = 'hyperparam_study'
    # fig_file += f'l=' + '+'.join(layers) + '-'
    # fig_file += f'd=' + '+'.join(emb_ds) + '-'
    # fig_file += f'qk_share' + '+'.join(qk_shares)
    #fig_file += '_'.join(model_types_short)+ '-' + metrics[0]'
    fig_file += '.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file))            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])