
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np
import pandas as pd
import json

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
#linestyles = ['solid', 'densely dashed', 'dashed', 'densely dotted', 'dotted']
#linestyles = ['-', '--', '-.', ':']
linestyles = ['-', '--', ':']
colors = list(mcl.TABLEAU_COLORS.keys())
# ------------------------------------------


# Plots average of metrics over ensembles
"""

"""

def plot_ensembles(model_root_dir, datasets=['iwslt14'],  
                   metrics=['train_loss','val_loss','val_bleu'], 
                   mod_rows=1,display=False):
    global df, df_setting, df_filtered, fig_file, axs
    global model_dir, config_dict, final_metrics, ensemble_metrics, metric_plot 
    global model_dirs, config, attn_setup   

    datasets = str_to_ls(datasets)
    metrics = str_to_ls(metrics)

    mod_rows = int(mod_rows)
    display = str_to_bool(display)

    model_root_dir = model_root_dir.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(model_root_dir) if 'translation' in dirname])

    print(f'Datasets: {datasets}')
    print(f'model_root_dir = {model_root_dir}')
    # prompt to reorder file names
    for dirname_idx, dirname in enumerate(dirnames):
        print(f'Index {dirname_idx}: {dirname}')
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

                    if 'acc' in metric.lower or 'bleu' in metric.lower():
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
                axs[idx,kdx].plot(df_filtered.loc[iit,'iter'], ensemble_mean[iit],
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
                best = max(final_metrics[metric]) if 'acc' in metric or 'bleu' in metric else min(final_metrics[metric])
                worst = min(final_metrics[metric]) if 'acc' in metric or 'bleu' in metric else max(final_metrics[metric])
                median, mean = np.median(final_metrics[metric]), np.mean(final_metrics[metric])
                print(f'best, median, mean, worst {metric}: {best}, {median}, {mean}, {worst}')

                # if 'acc' in metric or 'f1' in metric:
                #     ylims[kdx] = max(ylims[kdx], mean)
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
        en_layers, de_layers = int(config_dict['encoder_layers']), int(config_dict['decoder_layers'])
        heads, hidden = int(config_dict['heads']), int(config_dict['hidden'])
        fig_file = f'en_layers={en_layers}-de_layers={de_layers}-heads={heads}-hidden={hidden}-'
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
