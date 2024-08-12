import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np
import pandas as pd
import json

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ---------- Global plot settings ----------
font_type = {'family' : 'sans-serif'}
plt.rc('font', **font_type)
plt.rc('legend',fontsize=7)
#linestyles = ['solid', 'densely dashed', 'dashed', 'densely dotted', 'dotted']
#linestyles = ['-', '--', '-.', ':']
linestyles = ['-', '--', ':']
markers = ['s', 'D', 'd', 'v', '^', 'o', '.']
markersize = '3'
colors = list(mcl.TABLEAU_COLORS.keys())
OTHER_COLORS = ['m', 'dimgray']
OTHER_COLORS_DICT = {'sink'+MODEL_SUFFIX: OTHER_COLORS[0], 'dp'+MODEL_SUFFIX: OTHER_COLORS[1]}
# ------------------------------------------


# Ablation study on alphas
def plot_fns_ensembles(models_roots, fns_type='spopfnsvit', metrics='val_acc',
                       is_single=True, cbar_separate=True, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, dataset, df_model
    global model_types, model_info, epochs, ensembles

    models_roots = str2ls(models_roots)
    model_root_dirs = models_roots
    print(model_root_dirs)

    metrics = str2ls(metrics)    
    is_single = str2bool(is_single)
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

    if is_single:
        nrows = 1
        ncols = len(model_root_dirs)
    else:
        nrows = len(model_root_dirs)    
        ncols = len(epss)

    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)
    
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

                epochs = run_perf.loc[:,'iter'].astype(int) // int(model_info['steps_per_epoch'].item())
                #epochs = run_perf.loc[:,'epoch']

                #trans = 1 if qk_share else 0.5
                trans = HYP_TRANS(alpha)
                if (row_idx,col_idx) == (0,0):
                    im = ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                        #,marker=model_markers[model_name], markersize=markersize,
                                        #,label=model_legend)    
                else:
                    ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=c_hyp, alpha=trans)
                                        #,marker=model_markers[model_name], markersize=markersize,
                                        #,label=model_legend)                                    

                # if row_idx == nrows - 1:
                #     if len(epochs) > 50:
                #         ax.set_xticks(epochs[49::50])
                #         ax.set_xticklabels(epochs[49::50])
                #     else:
                #         ax.set_xticks(epochs)
                #         ax.set_xticklabels(epochs)
                if row_idx != nrows - 1:
                    ax.set_xticklabels([])                

                # col labels (bandwidth)
                if row_idx == 0:
                    ax.set_title(rf'$\varepsilon = {{{eps}}}$')
                # row labels (Q = K)
                if col_idx == ncols - 1:
                    title = r'$W_Q \neq W_K$' if not qk_share else r'$W_Q = W_K$'               
                    ax.text(1.2, 0.5, title, transform=(
                                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                                    va='center', rotation='vertical')  # fontsize='medium',                 

                # subplot labels
                ax.text(
                    0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
                        ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                    va='bottom', fontfamily='sans-serif')  # fontsize='medium',              

            if eps == 1:

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

                        epochs = run_perf.loc[:,'iter'].astype(int) // int(model_info['steps_per_epoch'])

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

                        epochs = run_perf.loc[:,'iter'].astype(int) // int(model_info['steps_per_epoch'])

                        trans = 1
                        ax.plot(epochs, run_perf.loc[:,metrics[0]], linestyle=lstyle_model, c=OTHER_COLORS[1], alpha=trans)                                            
                
                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

            ax.grid()
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
    # fig.text(0.05, 0.5, 'Eval accuracy', fontsize='medium', va='center', rotation='vertical')    
    fig.supxlabel('Epochs', fontsize='medium')
    fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    # Adjust layout
    #plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'vit-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += '-'.join(model_types_plotted)+'_' + f'ds={dataset_name_short}'
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
def fns_fix_eps(models_roots, fns_type='spopfns'+MODEL_SUFFIX, metrics='val_acc',
                include_others=False, display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dirs, subpath, dirnames, model_root_dirs
    global model_combo, model_combos
    global alphas, epss, DCT_ALL, model_info, model_df, run_perf, dataset, df_model
    global model_types, model_info, epochs, ensembles
    global other_final_epoch_metrics

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

                epochs = run_perf.loc[:,'iter'].astype(int) // int(model_info['steps_per_epoch'])             

                fns_final_epoch_metrics[0, eps_idx, alp_idx] = run_perf.loc[run_perf.index[-1],metrics[0]]                                            
                if 'acc' in metrics[0]:
                    fns_final_epoch_metrics[1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].max()  
                else:
                    fns_final_epoch_metrics[1, eps_idx, alp_idx] = run_perf.loc[:,metrics[0]].min()

            if eps_idx == 0:                
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

                        epochs = run_perf.loc[:,'iter'].astype(int) // int(model_info['steps_per_epoch'])

                        other_final_epoch_metrics[model_type] = run_perf.loc[run_perf.index[-1],metrics[0]]

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

                        epochs = run_perf.loc[:,'iter'].astype(int) // int(model_info['steps_per_epoch'])

                        other_final_epoch_metrics[model_type] = run_perf.loc[run_perf.index[-1],metrics[0]]                                        
                
                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)

        for col_idx in range(fns_final_epoch_metrics.shape[0]):
            ax = axs[row_idx, col_idx]
            for eps_idx, eps in tqdm(enumerate(epss)):
                ax.plot(alphas, fns_final_epoch_metrics[col_idx,eps_idx,:], label = rf'$\varepsilon$ = {eps}',
                        linestyle=f'--', marker=markers[eps_idx],markersize=2)

            if include_others:
                for model_type in other_final_epoch_metrics.keys():
                    ax.axhline(y=other_final_epoch_metrics[model_type], linestyle=LINESTYLE_DICT[model_type], c=OTHER_COLORS_DICT[model_type])

            if row_idx == 0:
                title = 'Final' if col_idx==0 else 'Best'
                ax.set_title(title)

            # row labels (Q = K)
            if col_idx == ncols - 1:
                title = r'$W_Q \neq W_K$' if not qk_share else r'$W_Q = W_K$'               
                ax.text(1.2, 0.5, title, transform=(
                                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                                va='center', rotation='vertical')  # fontsize='medium',                            

            # subplot labels
            ax.text(
                0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                va='bottom', fontfamily='sans-serif')  # fontsize='medium',     

            ax.grid()
            #ax.yaxis.grid(True)

            total_figs += 0

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
    axs[0,0].legend(bbox_to_anchor=(0.85, 1.4),
                    loc='best', ncol=ncol_legend, frameon=False)  

    # axs[0,0].legend(loc='upper left', ncol=1, frameon=False)  

    # Add shared x and y labels   
    fig.supxlabel(r'$\alpha$', fontsize='medium')
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

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'vit-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = f'fns_fix_eps-layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += f'ds={dataset_name_short}'
        # if isfile(njoin(SAVE_DIR, fig_file)):
        #     version = len([fname for fname in os.listdir(SAVE_DIR) if fname==fig_file])
        #     fig_file += f'-v{version}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file))            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')


# Plots average of metrics over ensembles
def plot_ensembles(model_root_dir, datasets=['cifar10'],  
                   metrics=['train_loss','val_loss','train_acc','val_acc'], 
                   mod_rows=1,display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dir, config_dict, final_metrics, ensemble_metrics, metric_plot 
    global model_dirs, config, attn_setup   

    datasets = str_to_ls(datasets)
    metrics = str_to_ls(metrics)

    mod_rows = int(mod_rows)
    display = str_to_bool(display)

    model_root_dir = model_root_dir.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(model_root_dir) if 'vit' in dirname])

    print(f'Datasets: {datasets}')
    print(f'model_root_dir = {model_root_dir}')
    # prompt to reorder file names
    for dirname_idx, dirname in enumerate(dirnames):
        for subdir in os.listdir(njoin(model_root_dir, dirname)):
            if isfile(njoin(model_root_dir, dirname, subdir, 'run_performance.csv')):
                print(f'Index {dirname_idx}: {dirname}')
                break
    dirname_idxs = input('Order of dirnames:')
    dirname_idxs = [int(dirname_idx) for dirname_idx in dirname_idxs.split(',')]
    assert len(dirname_idxs) <= len(dirnames), 'dirname_idxs cannot exceed dirnames'
    dirnames = [dirnames[dirname_idx] for dirname_idx in dirname_idxs]
    print(f'{metrics} \n')

    nrows, ncols = len(datasets), len(metrics)
    #figsize = (9.5,2.5*nrows)
    figsize = (12,2.5*nrows)
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
    for idx, dataset in tqdm(enumerate(datasets)):
        print('\n' + '#'*25 + f' Training on {dataset} '  + '#'*25 + '\n')

        model_names = []
        model_types = {}
        model_linestyles = {}
        model_colors = {}
        N_model_types = 0        
        for jdx, dirname in enumerate(dirnames):             
            ensemble_dir = njoin(model_root_dir,dirname)
            model_dirs = []
            for subdir in os.listdir(ensemble_dir):
                subpath = njoin(ensemble_dir,subdir)
                if 'model=' in subdir and isfile(njoin(subpath,'run_performance.csv')):
                    model_dirs.append(subpath.replace('\\',''))

            # get type of transformer
            #df_setting = pd.read_csv(njoin(model_dirs[0],'train_setting.csv'))

            with open(njoin(model_dirs[0], 'config.json')) as json_file:
                config = json.load(json_file)
            with open(njoin(model_dirs[0], 'attn_setup.json')) as json_file:
                attn_setup = json.load(json_file)         
            model_type = model_name = NAMES_DICT[dirname.split('-')[0]]
            if model_type not in model_types.keys():                
                model_types[model_name] = 1
                N_model_types += 1
            else:
                model_types[model_name] += 1            
            if model_name not in model_names:
                model_names.append(model_name)
            if 'fns' in dirname.lower():
                alpha, bandwidth, a  = attn_setup['alpha'], attn_setup['bandwidth'], attn_setup['a']
                #model_settings = rf'$\alpha$ = {alpha}, $\varepsilon$ = {bandwidth}'
                model_settings = rf'$\alpha$ = {alpha}, $a$ = {a}'
                # if alpha < 2:                        
                #     d_intrinsic = attn_setup.loc[0,'d_intrinsic']
                #     model_settings += rf', $d$ = {d_intrinsic}'  #$d_{\mathcal{M}}$
                model_name += f' ({model_settings})'
            elif 'sink' in dirname.lower():
                n_it  = attn_setup['n_it']
                model_settings = rf'iter = {n_it}'
                model_name += f' ({model_settings})'
            model_linestyles[model_name] = linestyles[model_types[model_type] - 1]
            model_colors[model_name] = colors[N_model_types - 1]

            # ensemble of training instances for the same architecture
            ylims = [100,100,0,0]
            count = 0
            final_metrics = {}  # metrics of the final epoch            
            ensemble_metrics = {}            
            for model_dir in model_dirs:
                df = pd.read_csv(njoin(model_dir, 'run_performance.csv'), index_col=0)     
                                   
                for kdx, metric in enumerate(metrics):
                    df_filtered = df[df[metric].notna()]        
                    df_filtered = df_filtered[::mod_rows]            

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

                    # epoch_eval_runtime = df[df['eval_runtime'].notna()].loc[:,'eval_runtime'].mean()
                    # train_runtime, total_flos = df_setting.loc[0,['train_runtime', 'total_flos']]
                    # for other_metric in ['epoch_eval_runtime', 'train_runtime', 'total_flos']:
                    #     if other_metric not in final_metrics.keys():
                    #         final_metrics[other_metric] = [locals()[other_metric]]
                    #     else:
                    #         final_metrics[other_metric].append(locals()[other_metric])
                
                count += 1
            final_metrics[f'count'] = count                        
                        
            print('-'*15)                
            print(f'{model_name}, total ensembles = {len(model_dirs)}')
            for kdx, metric in enumerate(metrics):

                ensemble_metrics[metric] = pd.concat(ensemble_metrics[metric], axis=1).T
                # quit()

                # ----- Plots -----
                ensemble_mean = ensemble_metrics[metric].mean(0)
                ensemble_std = ensemble_metrics[metric].std(0)
                max_iter = df_filtered.loc[:,'iter'].max()

                iit = df_filtered[df_filtered.loc[:,'iter']>=max_iter*1/3].index
                # df_filtered.loc[iit,'iter'], ensemble_mean[iit],
                axs[idx,kdx].plot(df_filtered.index, ensemble_mean,
                                  linestyle=model_linestyles[model_name], c=model_colors[model_name],
                                  label=model_name)                 

                # std
                # axs[idx,kdx].fill_between(df_filtered.loc[:,'epoch'], 
                #                           ensemble_mean - ensemble_std, ensemble_mean + ensemble_std, 
                #                           color=model_colors[model_name], alpha=0.5)                                          

                if idx == 0:
                    axs[idx,kdx].set_title(NAMES_DICT[metric])
                #if idx == nrows - 1:
                axs[0,kdx].set_xlabel('Steps')
                axs[idx,kdx].ticklabel_format(style='sci',scilimits=(-3,10),axis='x')

                #axs[idx,0].set_ylabel(NAMES_DICT[dataset])
                axs[0,0].legend(loc='upper right', #bbox_to_anchor=(0.5, 1.05),
                                ncol=1, frameon=False)    

                # ----- Messages -----
                best = max(final_metrics[metric]) if 'acc' in metric or 'f1' in metric else min(final_metrics[metric])
                worst = min(final_metrics[metric]) if 'acc' in metric or 'f1' in metric else max(final_metrics[metric])
                median, mean = np.median(final_metrics[metric]), np.mean(final_metrics[metric])
                print(f'best, median, mean, worst {metric}: {best}, {median}, {mean}, {worst}')

                if 'acc' in metric or 'f1' in metric: 
                    if 'val' in metric:                   
                        axs[idx,kdx].set_ylim([40, 65])
                    else:
                        axs[idx,kdx].set_ylim([40, 90])
                # else:
                #     ylims[kdx] = min(ylims[kdx], mean)
                # axs[idx,kdx].set_ylim

            # for other_metric in ['epoch_eval_runtime', 'train_runtime', 'total_flos']:
            #     print(f'Average total {other_metric}: {np.mean(final_metrics[other_metric])}')
            print('-'*15 + '\n')                                


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


# Example:
"""
python -i plot_results.py plot_model .droot/formers_trained
"""
def plot_model(model_root_dir, dirnames, instances, 
               datasets, metrics=['train_loss','val_loss','train_acc','val_acc'], 
               mod_rows=1,
               display=False):
    
    """
    - model_root_dir (str): the root dir of models performance saved
    - dirnames (str): the dirnames of the models
    - instances (str): the instance of training
    - mod_rows (int): number for modding df_filtered
    - display (bool): whether to display the figure
    """

    global df, df_setting, df_filtered, fig_file, axs
    global model_dir
    global config_dict, model_dict
    global metric_plot
    # for local_keys in ['dirnames', 'datasets', 'metrics']:
    #     locals()[local_keys] = str_to_ls(locals()[local_keys])

    dirnames = str_to_ls(dirnames)
    datasets = str_to_ls(datasets)
    instances = str_to_ls(instances)
    metrics = str_to_ls(metrics)
    display = str_to_bool(display)

    mod_rows = int(mod_rows)

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
    qk_share = 'qqv' if 'qqv' in model_root_dir else 'qkv'
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
            #df_setting = pd.read_csv(njoin(model_dir,'final_performance.csv'))  
            print_metrics = {}      
            for kdx, metric in enumerate(metrics):
                df_filtered = df[df[metric].notna()]
                df_filtered = df_filtered.iloc[::mod_rows]

                model_name = NAMES_DICT[dirname.split('-')[0]]
                if model_name not in model_names:
                    model_names.append(model_name)
                if 'fns' in dirname:
                    #alpha, bandwidth  = df_setting.loc[0,['alpha','bandwidth']]    
                    model_dict = {}
                    for ls in dirname.split('/'):
                        for ele in ls.split('-'):
                            if '=' in ele:
                                key, val = ele.split('=')
                                model_dict[key] = val

                    alpha, bandwidth = model_dict['alpha'], model_dict['eps']
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

                axs[idx,kdx].plot(df_filtered.loc[:,'iter'], metric_plot, label=model_name,
                                  linewidth=lwidth)

                if idx == 0:
                    axs[idx,kdx].set_title(NAMES_DICT[metric])
                if idx == nrows - 1:
                    axs[idx,kdx].set_xlabel('Steps')

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