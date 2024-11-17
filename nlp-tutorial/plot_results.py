import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import math
import numpy as np
import pandas as pd

from ast import literal_eval
from matplotlib.transforms import ScaledTranslation
from os import makedirs
from os.path import isdir, isfile
from string import ascii_lowercase
from time import time
from tqdm import tqdm
from constants import *
from mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history
from mutils import collect_model_dirs

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
OTHER_COLORS = ['m', 'dimgray']
OTHER_COLORS_DICT = {'sink'+MODEL_SUFFIX: OTHER_COLORS[0], 'dp'+MODEL_SUFFIX: OTHER_COLORS[1],
                     'opsink'+MODEL_SUFFIX: OTHER_COLORS[0], 'opdp'+MODEL_SUFFIX: OTHER_COLORS[1],}
# ------------------------------------------

# Plots average of metrics over ensembles
"""
python -i plot_results.py plot_ensembles .droot/formers_trained/layers=2-heads=8-hidden=768-epochs=10-qkv/
PROMPT input:
"""

# Ablation study on alphas
def fns_ensembles(models_roots, fns_type='spopfns'+MODEL_SUFFIX, metrics='eval_accuracy',
                       cbar_separate=True, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos, model_types_plotted, model_types_short
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, dataset, df_model
    global model_types, model_info, epochs, ensembles

    models_roots = str2ls(models_roots)
    model_root_dirs = models_roots
    print(model_root_dirs)

    metrics = str2ls(metrics)    
    display = str2bool(display)    
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
    num_attention_heads, num_hidden_layers, hidden_size = df_model.loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]
    dataset = df_model.loc[0,'dataset_name']

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    epss = sorted(df_model.loc[:,'bandwidth'].unique())    

    nrows = len(model_root_dirs)    
    ncols = len(epss)

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
    for row_idx, models_root in enumerate(models_roots):
        DCT_ALL = collect_model_dirs(models_root)
        if DCT_ALL == {}:
            continue
        for col_idx, eps in tqdm(enumerate(epss)):
            ax = axs[row_idx, col_idx]

            model_type = fns_type
            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            model_df = DCT_ALL[model_type].dropna(subset='alpha')
            model_df.reset_index(drop=True, inplace=True)
            lstyle_model = LINESTYLE_DICT[model_type]

            # -------------------- FNS --------------------
            for alpha in alphas:
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & (model_df['ensembles']>0)]
                if len(model_info.index) == 0:
                    continue

                instances = model_info['instances'].item()
                qk_share = model_info['qk_share'].item()
                
                model_instance_path = njoin(model_info['model_dir'].item(), f'model={instances[0]}')
                if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                    run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                    run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))

                epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int))                

                trans = 1
                #trans = HYP_TRANS(alpha)
                if (row_idx,col_idx) == (0,0):
                    im = ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                        #,marker=model_markers[model_name], markersize=markersize,
                                        #,label=model_legend)    
                else:
                    ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                        #,marker=model_markers[model_name], markersize=markersize,
                                        #,label=model_legend)                                    

                # if row_idx != nrows - 1:
                #     ax.set_xticklabels([])

                # col labels (bandwidth)
                if row_idx == 0:
                    ax.set_title(rf'$\varepsilon = {{{eps}}}$')
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
                    va='bottom', fontfamily='sans-serif')  # fontsize='medium',              

            #if eps == 1:                
            if col_idx == ncols - 1:
            # -------------------- SINK --------------------                
                model_type = 'sink' + suffix
                if model_type in model_types:
                    model_df = DCT_ALL[model_type]
                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = model_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    instances = model_info['instances']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:
                        model_instance_path = njoin(model_info['model_dir'], f'model={instances[0]}')
                        if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                        if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))

                        epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int)) 

                        trans = 1
                        ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=OTHER_COLORS[0], alpha=trans)    

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

            # -------------------- DP --------------------

                model_type = 'dp' + suffix
                if model_type in model_types:       
                    model_df = DCT_ALL[model_type]
                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = model_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    instances = model_info['instances']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:
                        model_instance_path = njoin(model_info['model_dir'], f'model={instances[0]}')
                        if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                        if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                            run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))                        

                        epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int)) 

                        trans = 1
                        ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=OTHER_COLORS[1], alpha=trans)                                            
                
                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

            ax.grid()
            #ax.yaxis.grid(True)
            total_figs += 1

    # legend
    # for alpha in alphas:
    #     axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
    #                 label=rf'$\alpha$ = {alpha}')    

    for model_type in model_types_plotted:   
        if 'fns' in model_type:
            color = 'k'
        elif 'sink' in model_type:
            color = OTHER_COLORS[0]
        elif 'dp' in model_type:
            color = OTHER_COLORS[1]
        axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], 
                    label=NAMES_DICT[model_type])

    ncol_legend = 2 if len(model_types_plotted) == 3 else 1
    if len(model_types_plotted) >= 2:
        #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
        axs[0,0].legend(bbox_to_anchor=(0.85, 1.35),
                    loc='best', ncol=ncol_legend, frameon=False)                    

    # Add shared x and y labels
    # fig.text(0.5, 0.01, 'Epochs', fontsize='medium', ha='center')
    # fig.text(0, 0.5, NAMES_DICT[metrics[0]], fontsize='medium', va='center', rotation='vertical')       
    fig.supxlabel('Epochs', fontsize='medium')
    fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'
        fig_file += f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
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


# Ablation study on bandwidth
def fns_fix_eps(models_roots, fns_type='spopfns'+MODEL_SUFFIX, metrics='eval_accuracy',
                include_others=False, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, dataset, df_model
    global model_types, model_info, epochs, ensembles
    global other_final_epoch_metrics

    MARKERSIZE = 4
    BIGGER_SIZE = 10
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title    

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
    num_attention_heads, num_hidden_layers, hidden_size = df_model.loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]
    dataset = df_model.loc[0,'dataset_name']

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    epss = sorted(df_model.loc[:,'bandwidth'].unique())    

    nrows, ncols = len(models_roots), 2

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
    for row_idx, models_root in enumerate(models_roots):
        fns_final_epoch_metrics = np.zeros([2, len(epss), len(alphas)])
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
            for alp_idx, alpha in enumerate(alphas):
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & (model_df['ensembles']>0)]
                if len(model_info.index) == 0:
                    continue

                instances = model_info['instances'].item()
                qk_share = model_info['qk_share'].item()
                
                model_instance_path = njoin(model_info['model_dir'].item(), f'model={instances[0]}')
                if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                    run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                    run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))

                if 'acc' in metrics[0]:
                    run_perf.loc[:,metrics[0]] *= 100                     
                epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int))                

                fns_final_epoch_metrics[0, eps_idx, alp_idx] = run_perf.loc[run_perf.index[-1],metrics[0]]                                            
                if 'acc' in metrics[0]:
                    fns_final_epoch_metrics[1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].max()  
                else:
                    fns_final_epoch_metrics[1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].min()

            if eps_idx == 0:                
            # -------------------- SINK, DP --------------------                
                other_model_types = ['sink' + suffix, 'dp' + suffix]
                for model_type in other_model_types:
                    if model_type in model_types:
                        model_df = DCT_ALL[model_type]
                        lstyle_model = LINESTYLE_DICT[model_type]

                        model_info = model_df.iloc[0,:]
                        ensembles = model_info['ensembles']
                        instances = model_info['instances']
                        qk_share = model_info['qk_share']
                        if ensembles > 0:
                            model_instance_path = njoin(model_info['model_dir'], f'model={instances[0]}')
                            if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                            if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))

                            if 'acc' in metrics[0]:
                                run_perf.loc[:,metrics[0]] *= 100      
                            epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int)) 

                            other_final_epoch_metrics[model_type] = run_perf.loc[run_perf.index[-1],metrics[0]]
                            if 'acc' in metrics[0]:
                                other_best_epoch_metrics[model_type] = run_perf.loc[:,metrics[0]].max()  
                            else:
                                other_best_epoch_metrics[model_type] = run_perf.loc[:,metrics[0]].min()

                        if model_type not in model_types_plotted:
                            model_types_plotted.append(model_type)

        for col_idx in range(fns_final_epoch_metrics.shape[0]):
            ax = axs[row_idx, col_idx]
            for eps_idx, eps in tqdm(enumerate(epss)):
                ax.plot(alphas, fns_final_epoch_metrics[col_idx,eps_idx,:], label = rf'$\varepsilon$ = {eps}',
                        linestyle=f'--', marker=markers[eps_idx],markersize=MARKERSIZE)

            if include_others:
                for model_type in other_final_epoch_metrics.keys():
                    if col_idx == 0:
                        ax.axhline(y=other_final_epoch_metrics[model_type], 
                                   linestyle=LINESTYLE_DICT[model_type], c=OTHER_COLORS_DICT[model_type])
                    else:
                        ax.axhline(y=other_best_epoch_metrics[model_type], 
                                   linestyle=LINESTYLE_DICT[model_type], c=OTHER_COLORS_DICT[model_type])

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

            ax.grid()
            #ax.yaxis.grid(True)
            ax.set_xticks(alphas)

            total_figs += 1

    

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
        fig_file = models_root.split('/')[1] + '-'
        fig_file += f'fns_fix_eps-layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += f'ds={dataset_name_short}'
        # if isfile(njoin(SAVE_DIR, fig_file)):
        #     version = len([fname for fname in os.listdir(SAVE_DIR) if fname==fig_file])
        #     fig_file += f'-v{version}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')


# assumption 1 and 2 possibilities for full-sized models
def phase_ensembles(models_root, fns_manifold='sphere', metrics='val_acc,val_loss',
                    is_ops = [False, True],
                    cbar_separate=False, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos, model_types_plotted, model_types_short
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, run_perf_all, dataset, df_model
    global model_types, model_info, epochs, ensembles, instances
    global models_roots, qk_shares, matching_df
    global model_type, qk_share
    global DCT_cur, df_model_cur, other_matching_df
    global model_instance_path    

    assert fns_manifold in ['sphere', 'rd'], f'{fns_manifold} does not exist!'
    manifold_prefix = 'sp' if fns_manifold == 'sphere' else 'rd'

    BIGGER_SIZE = 10
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

    #df_model = DCT_ALL['spopfnsformer']
    df_model = DCT_ALL['rdfnsformer']
    #df_model.reset_index(drop=True, inplace=True)
    
    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,['n_heads', 'n_layers', 'hidden']]
    dataset = DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,'dataset_name']

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    epss = sorted(df_model.loc[:,'bandwidth'].unique())    

    #alpha = alphas[-1]
    eps = epss[0]

    #metric = metrics[0]
    qk_shares = list(df_model.loc[:,'qk_share'].unique())    
    #is_ops = [False]
    #nrows = len(model_root_dirs)    
    nrows = len(qk_shares)
    ncols = len(is_ops)

    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=False)  # layout='constrained'
    
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)                 

    model_types_plotted = []
    model_types_instances = {}
    #for row_idx, models_root in enumerate(models_roots):    
    qk_share = False
    #for row_idx, qk_share in enumerate(qk_shares):        
    for row_idx, metric in enumerate(metrics):
        for col_idx, is_op in enumerate(is_ops):

            ax = axs[row_idx, col_idx]
            model_type = 'op' + manifold_prefix + 'fns' + MODEL_SUFFIX if is_op else manifold_prefix + 'fns' + MODEL_SUFFIX
            df_model = DCT_ALL[model_type]
            matching_df = df_model[(df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&
                                   (df_model['is_op']==is_op)&
                                   (df_model['model_dir'].str.contains(model_type))
                                   #(df_model['model_dir']==model_type)
                                   ]
            #print(matching_df)

            if matching_df.shape[0] > 0: 
                DCT_matching = { model_type: matching_df }
            else:
                DCT_matching = {}

            if DCT_matching == {}:
                continue

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            for alpha in alphas:
            #for alpha in alphas[-1:]:
                model_df = DCT_matching[model_type].dropna(subset='alpha')
                model_df.reset_index(drop=True, inplace=True)
                lstyle_model = LINESTYLE_DICT[model_type]

                # -------------------- FNS --------------------                        
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & 
                                      (model_df['ensembles']>0)]
                if len(model_info.index) == 0:
                    continue

                instances = model_info['instances'].item()
                qk_share = model_info['qk_share'].item()
                
                run_perf_all = []
                for instance in instances:
                    model_instance_path = njoin(model_info['model_dir'].item(), f'model={instance}')
                    if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                    if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))            

                    epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int))                
                    if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                        run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100

                    run_perf_all.append(run_perf.loc[:,metric])

                run_perf_all = pd.concat(run_perf_all, axis=1)

                metric_m = run_perf_all.quantile(0.5,1)
                metric_l = run_perf_all.quantile(0.25,1)
                metric_u = run_perf_all.quantile(0.75,1)

                trans = 1
                #trans = HYP_TRANS(alpha)
                if (row_idx,col_idx) == (0,0):
                    im = ax.plot(epochs, metric_m, linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                            
                    ax.fill_between(epochs, metric_l, metric_u,
                                    color=c_hyp, alpha=trans/2)
                else:
                    ax.plot(epochs, metric_m, linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                         
                    ax.fill_between(epochs, metric_l, metric_u,
                                    color=c_hyp, alpha=trans/2)                                                                        

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
                if model_type in model_types:
                    model_df = DCT_ALL[model_type]
                    other_matching_df = model_df[(model_df['ensembles']>0)&(model_df['qk_share']==qk_share)&
                                                 (model_df['is_op']==is_op)&
                                                 (model_df['model_dir'].str.contains(model_type))]                

                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = other_matching_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    instances = model_info['instances']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:

                        run_perf_all = []
                        for instance in instances:
                            model_instance_path = njoin(model_info['model_dir'], f'model={instance}')
                            if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                            if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))

                            epochs = run_perf.loc[:,'iter'].astype(int) // int(run_perf.loc[0,'iter'].astype(int)) 
                            if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                                run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100
                        
                            run_perf_all.append(run_perf.loc[:,metric])
                        run_perf_all = pd.concat(run_perf_all, axis=1)

                        metric_m = run_perf_all.quantile(0.5,1)
                        metric_l = run_perf_all.quantile(0.25,1)
                        metric_u = run_perf_all.quantile(0.75,1)
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

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

            #ax.grid(axis='y')
            ax.grid()
            #ax.axvline(x=15, color='k', linestyle='--', linewidth=0.8)
            #ax.yaxis.grid(True)        

    #axs[0,0].set_ylim([75,85])        
    #axs[0,0].set_ylim([79,84])
    #axs[0,0].set_ylim([75,85])
    axs[0,0].set_ylim([75,84])
    axs[1,0].set_ylim([0.45,0.65])
    axs[1,0].set_xticks([5,10,15,20])

    # labels
    model_labels = []
    for model_type in model_types_plotted:   
        if 'op' in model_type:
            model_type = model_type.replace('op', '')

        if 'fns' in model_type:
            color = 'k'
        elif 'sink' in model_type:
            color = OTHER_COLORS[0]
        elif 'dp' in model_type:
            color = OTHER_COLORS[1]
            
        model_label = NAMES_DICT[model_type]
        if model_label not in model_labels:            
            axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], 
                          label=model_label)

            model_labels.append(model_label)

    # legend
    for alpha in alphas[::-1]:
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
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'
        #fig_file += f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += f'layers={num_hidden_layers}-hidden={hidden_size}-'
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


# compare full-sized models
def full_ensembles(models_roots, fns_type='spopfns'+MODEL_SUFFIX, metrics='eval_accuracy,eval_loss',
                  cbar_separate=False, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos, model_types_plotted, model_types_short
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, run_perf_all, dataset, df_model
    global model_types, model_info, epochs, ensembles, instances

    BIGGER_SIZE = 10
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    models_roots = str2ls(models_roots)
    model_root_dirs = models_roots
    print(model_root_dirs)

    metrics = str2ls(metrics)    
    display = str2bool(display)    

    suffix = fns_type.split('fns')[-1]
    DCT_ALL = collect_model_dirs(model_root_dirs[0], suffix=suffix)
    model_types = list(DCT_ALL.keys())
    for model_type in model_types:
        if 'fns' in model_type:
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            df_model.reset_index(drop=True, inplace=True)
            break

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = df_model.loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]
    dataset = df_model.loc[0,'dataset_name']

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    epss = sorted(df_model.loc[:,'bandwidth'].unique())    

    #alpha = alphas[-1]
    eps = epss[0]

    nrows = len(model_root_dirs)    
    ncols = len(metrics)

    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=False)  # layout='constrained'
    
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)     
        
    model_types_plotted = []
    model_types_instances = {}
    for row_idx, models_root in enumerate(models_roots):
        DCT_ALL = collect_model_dirs(models_root)
        if DCT_ALL == {}:
            continue
        for col_idx, metric in enumerate(metrics):

            ax = axs[row_idx, col_idx]

            model_type = fns_type
            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            for alpha in alphas:
            #for alpha in alphas[-1:]:
                model_df = DCT_ALL[model_type].dropna(subset='alpha')
                model_df.reset_index(drop=True, inplace=True)
                lstyle_model = LINESTYLE_DICT[model_type]

                # -------------------- FNS --------------------                        
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color  

                model_info = model_df[(model_df['alpha']==alpha) & (model_df['bandwidth']==eps) & (model_df['ensembles']>0)]
                if len(model_info.index) == 0:
                    continue

                instances = model_info['instances'].item()
                qk_share = model_info['qk_share'].item()
                
                run_perf_all = []
                for instance in instances:
                    model_instance_path = njoin(model_info['model_dir'].item(), f'model={instance}')
                    if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                    if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                        run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))            

                    epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int))                
                    if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                        run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100

                    run_perf_all.append(run_perf.loc[:,metric])

                run_perf_all = pd.concat(run_perf_all, axis=1)
                metric_mean = run_perf_all.mean(1)
                metric_std = run_perf_all.std(1)

                trans = 1
                #trans = HYP_TRANS(alpha)
                if (row_idx,col_idx) == (0,0):
                    im = ax.plot(epochs, metric_mean, linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                        #,marker=model_markers[model_name], markersize=markersize,
                                        #,label=model_legend)   
                                            
                    ax.fill_between(epochs, metric_mean-metric_std, metric_mean+metric_std,
                                    color=c_hyp, alpha=trans/2)
                else:
                    ax.plot(epochs, metric_mean, linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                        #,marker=model_markers[model_name], markersize=markersize,
                                        #,label=model_legend)
                                         
                    ax.fill_between(epochs, metric_mean-metric_std, metric_mean+metric_std,
                                    color=c_hyp, alpha=trans/2)                                                                        

            # if row_idx != nrows - 1:
            #     ax.set_xticklabels([])

            # col labels (bandwidth)
            # if row_idx == 0:
            #     ax.set_title(rf'$\varepsilon = {{{eps}}}$')
            # row labels (Q = K)
            if col_idx == ncols - 1:
                title = r'$Q \neq K$' if not qk_share else r'$Q = K$'               
                ax.text(1.2, 0.5, title, transform=(
                                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                                va='center', rotation='vertical')  # fontsize='medium',                          

            # -------------------- SINK, DP --------------------                
            #other_model_types = ['sink' + suffix, 'dp' + suffix]
            other_model_types = ['dp' + suffix]
            for oidx, model_type in enumerate(other_model_types):
                if model_type in model_types:
                    model_df = DCT_ALL[model_type]
                    lstyle_model = LINESTYLE_DICT[model_type]

                    model_info = model_df.iloc[0,:]
                    ensembles = model_info['ensembles']
                    instances = model_info['instances']
                    qk_share = model_info['qk_share']
                    if ensembles > 0:

                        run_perf_all = []
                        for instance in instances:
                            model_instance_path = njoin(model_info['model_dir'], f'model={instance}')
                            if isfile(njoin(model_instance_path, 'run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_instance_path, 'run_performance.csv'))
                            if isfile(njoin(model_instance_path, '_run_performance.csv')): 
                                run_perf = pd.read_csv(njoin(model_instance_path, '_run_performance.csv'))

                            epochs = run_perf.loc[:,'step'].astype(int) // int(run_perf.loc[0,'step'].astype(int)) 
                            if 'acc' in metric and run_perf.loc[run_perf.index[-1],metric] <= 1:
                                run_perf.loc[:,metric] = run_perf.loc[:,metric] * 100
                        
                            run_perf_all.append(run_perf.loc[:,metric])
                        run_perf_all = pd.concat(run_perf_all, axis=1)

                        metric_mean = run_perf_all.mean(1)
                        metric_std = run_perf_all.std(1)
                        trans = 1

                        ax.plot(epochs, metric_mean, linestyle=lstyle_model, 
                                c=OTHER_COLORS_DICT[model_type], alpha=trans)    
                                            
                        ax.fill_between(epochs, metric_mean-metric_std, metric_mean+metric_std,
                                        color=OTHER_COLORS_DICT[model_type], alpha=trans/2)    

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

            ax.grid()
            #ax.yaxis.grid(True)        

    for model_type in model_types_plotted:   
        if 'fns' in model_type:
            color = 'k'
        elif 'sink' in model_type:
            color = OTHER_COLORS[0]
        elif 'dp' in model_type:
            color = OTHER_COLORS[1]
        axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], 
                    label=NAMES_DICT[model_type])

    # legend
    for alpha in alphas[::-1]:
        axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
                      label=rf'$\alpha$ = {alpha}')   

    #ncol_legend = 2 if len(model_types_plotted) == 3 else 1
    ncol_legend = 2
    if len(model_types_plotted) >= 2:
        #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
        axs[0,0].legend(bbox_to_anchor=(0.85, 1.35),   # bbox_to_anchor=(0.85, 1.35)
                        loc='best', ncol=ncol_legend, frameon=False)                     

    # Add shared x and y labels     
    total_figs = 0      
    #fig.supxlabel('Epochs')  # , fontsize='medium'
    #fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')
    for row_idx in range(len(models_roots)):
        for col_idx, metric in enumerate(metrics):  
            ax = axs[row_idx, col_idx]
            #ax.set_ylabel(NAMES_DICT[metric])
            if row_idx == 0:
                ax.set_title(NAMES_DICT[metric])

            # subplot labels
            ax.text(
                0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom')  # , fontfamily='sans-serif', fontsize='medium',     

            if row_idx != 0:
                ax.sharey(axs[0, col_idx])

            total_figs += 1

            axs[-1,col_idx].set_xlabel('Epochs')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'
        fig_file += f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
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


# ---------------------------------------- END -----------------------------------------------------

# Comparisons for all models
def plot_ensembles(model_root_dirs, metrics=['eval_accuracy'], 
                   legend=True, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global final_performance
    global model_dir, config_dict, final_metrics, ensemble_metrics, metric_plot   
    global model_dirs, subpath 

    model_root_dirs = str2ls(model_root_dirs)
    metrics = str2ls(metrics)
    legend = str2bool(legend)
    display = str2bool(display)    
    assert len(metrics) == 1

    if len(model_root_dirs) <= 3:
        nrows = 1
        ncols = len(model_root_dirs)
    else:
        nrows = 2
        ncols = math.ceil(len(model_root_dirs)/nrows)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)
    if nrows == 1:
        if ncols > 1:
            axs = np.expand_dims(axs, axis=0)
        else:
            axs = np.expand_dims(axs, axis=[0,1])     
    axs = axs.flatten()

    for mr_ii, model_root_dir in enumerate(model_root_dirs):

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
        
        print(f'Dataset: {dataset}, model_root_dir = {model_root_dir} \n')

        dirname_idxs = input('Order of dirnames:')
        dirname_idxs = [int(dirname_idx) for dirname_idx in dirname_idxs.split(',')]
        assert len(dirname_idxs) <= len(dirnames), 'dirname_idxs cannot exceed dirnames'
        dirnames = [dirnames[dirname_idx] for dirname_idx in dirname_idxs]
        print(f'{metrics} \n')            

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
        model_markers = {}
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
            if 'fns' in dirname:
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
                #model_settings = rf'$\alpha$ = {alpha}, $a$ = {a}'
                model_settings = rf'$\alpha$ = {alpha}'
                # if alpha < 2:                        
                #     d_intrinsic = df_setting.loc[0,'d_intrinsic']
                #     model_settings += rf', $d$ = {d_intrinsic}'  #$d_{\mathcal{M}}$
                model_legend = f'{model_name} ({model_settings})'
            elif 'sink' in dirname:
                n_it  = df_setting.loc[0,'n_it']
                model_settings = rf'iter = {n_it}'
                model_legend = f'{model_name} ({model_settings})'        

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

            full_model_name = model_name.lower() + 'former'
            if 'fns' in model_name:
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color
            else:
                c_hyp = HYP_CMAP(HYP_CNORM(1))
            print(f'{alpha}: {c_hyp}')  # delete
            lstyle_model = LINESTYLE_DICT[full_model_name]   

            for kdx, metric in enumerate(metrics):

                ensemble_metrics[metric] = pd.concat(ensemble_metrics[metric], axis=1).T
                # quit()                

                # ----- Plots -----
                ensemble_mean = ensemble_metrics[metric].mean(0)
                ensemble_std = ensemble_metrics[metric].std(0)                

                axs[mr_ii].plot(df_filtered.loc[:,'epoch'], ensemble_mean,
                                linestyle=lstyle_model, c=c_hyp,
                                #marker=model_markers[model_name], markersize=markersize,
                                label=model_legend) 

                # std
                # axs[mr_ii].fill_between(df_filtered.loc[:,'epoch'], 
                #                         ensemble_mean - ensemble_std, ensemble_mean + ensemble_std, 
                #                         color=c_model, alpha=0.5)                                          

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

        # legend  
        if mr_ii == 0:
            #axs[mr_ii].legend(loc='best', ncol=2, frameon=False)           
            axs[mr_ii].legend(bbox_to_anchor=(1, 1.3),
                              loc='best', ncol=2, frameon=False)
        # col labels (bandwidth eps)
        if mr_ii < nrows:            
            axs[mr_ii].set_title(rf'$\varepsilon = {{{bandwidth}}}$')
        # row labels (Q = K)
        if mr_ii % ncols == ncols-1:      
            title = r'$Q \neq K$' if qk_share == 'qkv' else r'$Q = K$'               
            axs[mr_ii].text(1.2, 0.5, title, transform=(
                            axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                            fontsize='medium', va='center', rotation='vertical')

        # subplot labels
        axs[mr_ii].text(
            0.0, 1.0, f'({ascii_lowercase[mr_ii]})', transform=(
                axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='medium', va='bottom', fontfamily='sans-serif')                               


    # Add shared x and y labels
    fig.text(0.5, 0.01, 'Epochs', fontsize='medium', ha='center')
    fig.text(0, 0.5, NAMES_DICT[metrics[0]], fontsize='medium',va='center', rotation='vertical')    

    # Adjust layout
    #plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = ''
    if '_' in dataset:
        for s in dataset.split('_'):
            dataset_name_short += s[0]
    else:
        dataset_name_short += dataset

    if display:
        plt.show()
    else:
        FIGS_DIR = njoin(FIGS_DIR, 'nlp-task')
        if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
        if len(config_dict.keys()) != 0:
            layers, heads, hidden = int(config_dict['layers']), int(config_dict['heads']), int(config_dict['hidden'])
            fig_file = f'layers={layers}-heads={heads}-hidden={hidden}-'            
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
                # elif idx == nrows - 1:
                #     axs[idx,kdx].set_xlabel('Epoch')

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
        FIGS_DIR = njoin(FIGS_DIR, 'nlp-task')
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