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

# Ablation study on alphas
def plot_fns_ensembles(model_root_dirs, metrics=['eval_accuracy'], 
                       epss=[0.01,0.1,1.0],
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
    assert len(model_root_dirs) % len(epss) == 0

    if len(model_root_dirs) <= 3:
        nrows = 1
        #ncols = len(model_root_dirs)
        ncols = len(epss)
    else:
        nrows = 2
        #ncols = math.ceil(len(model_root_dirs)/nrows)
        ncols = math.ceil(len(model_root_dirs)/nrows)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)
    if nrows == 1:
        if ncols > 1:
            axs = np.expand_dims(axs, axis=0)
        else:
            axs = np.expand_dims(axs, axis=[0,1])     
    axs = axs.flatten()

    alphas_plotted = []
    epss_plotted = []
    full_model_names = []
    for mr_ii, model_root_dir in enumerate(model_root_dirs):

        model_root_dir = model_root_dir.replace('\\','')
        #dirnames = sorted([dirname for dirname in os.listdir(model_root_dir) if 'former' in dirname])
        dirnames = sorted([dirname for dirname in os.listdir(model_root_dir) if 'fns' in dirname and 'former' in dirname])

        # Method 1: prompt to reorder file names
        # for dirname_idx, dirname in enumerate(dirnames):
        #     for subdir in os.listdir(njoin(model_root_dir, dirname)):
        #         if isfile(njoin(model_root_dir, dirname, subdir, 'run_performance.csv')):
        #             final_performance = pd.read_csv(njoin(model_root_dir, dirname, subdir, 'final_performance.csv'))
        #             dataset = final_performance.loc[0,'dataset_name']
        #             print(f'Index {dirname_idx}: {dirname}')
        #             break            
        # print(f'Dataset: {dataset}, model_root_dir = {model_root_dir} \n')
        # dirname_idxs = input('Order of dirnames:')        
        # dirname_idxs = [int(dirname_idx) for dirname_idx in dirname_idxs.split(',')]

        # Method 2: select based on eps
        eps_plot = epss[mr_ii % ncols]
        dirname_idxs = []
        for dirname_idx, dirname in enumerate(dirnames):
            if f'-eps={eps_plot}-' in dirname:
                for subdir in os.listdir(njoin(model_root_dir, dirname)):
                    if isfile(njoin(model_root_dir, dirname, subdir, 'run_performance.csv')):
                        final_performance = pd.read_csv(njoin(model_root_dir, dirname, subdir, 'final_performance.csv'))
                        dataset = final_performance.loc[0,'dataset_name']
                        print(f'Index {dirname_idx}: {dirname}')
                        dirname_idxs.append(dirname_idx)
                        break            
        print(f'Dataset: {dataset}, model_root_dir = {model_root_dir} \n')    

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

            #alpha, bandwidth  = df_setting.loc[0,['alpha','bandwidth']]      
            bandwidth = df_setting.loc[0,'bandwidth']
            if 'a' in df_setting.columns:
                a = df_setting.loc[0,'a']
            else:
                a = 0
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
                        
            train_setting = pd.read_csv(njoin(model_dirs[0], 'train_setting.csv'))
            if 'steps_per_train_epoch' not in train_setting.columns:
                steps_per_train_epoch = df_filtered.loc[0,'step']
            else:
                steps_per_train_epoch = train_setting.loc[0,'steps_per_train_epoch']

            print('-'*15)    
            print(f'{model_name} on {dataset}')

            full_model_name = model_name.lower() + 'former'
            c_hyp = HYP_CMAP(HYP_CNORM(alpha))  # hyperparameter color            
            lstyle_model = LINESTYLE_DICT[full_model_name]   

            if alpha not in alphas_plotted:
                alphas_plotted.append(alpha)
            if bandwidth not in epss_plotted:
                epss_plotted.append(bandwidth)
            if full_model_name not in full_model_names:
                full_model_names.append(full_model_name)

            for kdx, metric in enumerate(metrics):

                ensemble_metrics[metric] = pd.concat(ensemble_metrics[metric], axis=1).T
                # quit()                

                # ----- Plots -----
                ensemble_mean = ensemble_metrics[metric].mean(0)
                ensemble_std = ensemble_metrics[metric].std(0)                

                epochs = df_filtered.loc[:,'step'].astype(int) // int(steps_per_train_epoch)                
                # df_filtered.loc[:,'epoch']
                axs[mr_ii].plot(epochs, ensemble_mean,
                                linestyle=lstyle_model, c=c_hyp
                                #,marker=model_markers[model_name], markersize=markersize,
                                #,label=model_legend
                                ) 

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

        # col labels (bandwidth eps)
        if mr_ii < ncols:            
            axs[mr_ii].set_title(rf'$\varepsilon = {{{bandwidth}}}$')
        # row labels (Q = K)
        if mr_ii % ncols == ncols-1:      
            title = r'$W_Q \neq W_K$' if qk_share == 'qkv' else r'$W_Q = W_K$'               
            axs[mr_ii].text(1.2, 0.5, title, transform=(
                            axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                            va='center', rotation='vertical')  # fontsize='medium', 
        if mr_ii // ncols == nrows-1:
            axs[mr_ii].set_xticks(epochs)
            axs[mr_ii].set_xticklabels(epochs)

        # subplot labels
        axs[mr_ii].text(
            0.0, 1.0, f'({ascii_lowercase[mr_ii]})', transform=(
                axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            va='bottom', fontfamily='sans-serif')  # fontsize='medium',                                

    # legend
    for alpha_plotted in alphas_plotted :
        axs[0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha_plotted)), linestyle='solid', 
                    label=rf'$\alpha$ = {alpha_plotted}')    
    for full_model_name in full_model_names:        
        axs[0].plot([], [], c='k', linestyle=LINESTYLE_DICT[full_model_name], 
                    label=NAMES_DICT[full_model_name])
    #axs[0].legend(loc='best', ncol=2, frameon=False)           
    axs[0].legend(bbox_to_anchor=(0.85, 1.35),
                  loc='best', ncol=2, frameon=False)                    

    # Add shared x and y labels
    fig.text(0.5, 0.01, 'Epochs', fontsize='medium', ha='center')
    fig.text(0.05, 0.5, 'Eval accuracy', fontsize='medium', va='center', rotation='vertical')    

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
        from constants import FIGS_DIR
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
            title = r'$W_Q \neq W_K$' if qk_share == 'qkv' else r'$W_Q = W_K$'               
            axs[mr_ii].text(1.2, 0.5, title, transform=(
                            axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                            fontsize='medium', va='center', rotation='vertical')

        # subplot labels
        axs[mr_ii].text(
            0.0, 1.0, f'({ascii_lowercase[mr_ii]})', transform=(
                axs[mr_ii].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='medium', va='bottom', fontfamily='sans-serif')                               


    # Add shared x and y labels
    fig.text(0.5, 0.01, 'Epochs', ha='center')
    fig.text(0, 0.5, 'Eval accuracy', va='center', rotation='vertical')    

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