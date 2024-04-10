import matplotlib.pyplot as plt
import pandas as pd

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
# ------------------------------------------

"""
python -i plot_results.py plot_model .droot/trained_models_v2/ vsfnsformer,dpformer 0,1 rotten_tomatoes,imdb eval_loss,eval_accuracy,eval_f1_score
"""
def plot_model(model_root_dir, model_names, instances, 
               datasets, metrics, display=False):
    global df, df_filtered, fig_file

    # for local_keys in ['model_names', 'datasets', 'metrics']:
    #     locals()[local_keys] = str_to_ls(locals()[local_keys])

    model_names = str_to_ls(model_names)
    datasets = str_to_ls(datasets)
    instances = str_to_ls(instances)
    metrics = str_to_ls(metrics)
    display = str_to_bool(display)

    print(metrics)

    nrows, ncols = len(datasets), len(metrics)
    figsize = (10,5)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=True,sharey=False)
    for idx, dataset in tqdm(enumerate(datasets)):
        for jdx, model_name in enumerate(model_names): 
            model_dir = njoin(model_root_dir, f'{model_name}_{dataset}', f'model={instances[jdx]}')
            df = pd.read_csv(njoin(model_dir, 'run_performance.csv'))            
            for kdx, metric in enumerate(metrics):
                df_filtered = df[df[metric].notna()]

                label = NAMES_DICT[model_name]
                if 'acc' in metric or 'f1' in metric:
                    metric_plot = df_filtered.loc[:,metric] * 100
                else:
                    metric_plot = df_filtered.loc[:,metric]
                axs[idx,kdx].plot(df_filtered.loc[:,'epoch'], metric_plot, label=label)

                if idx == 0:
                    axs[idx,kdx].set_title(NAMES_DICT[metric])
                elif idx == nrows - 1:
                    axs[idx,kdx].set_xlabel('Epoch')

        axs[idx,0].set_ylabel(NAMES_DICT[dataset])
    axs[0,0].legend(loc=7)

    if display:
        plt.show()
    else:
        if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
        fig_file = '-'.join(model_names)+'_'+'-'.join(datasets)+'.pdf'
        plt.savefig(njoin(FIGS_DIR, fig_file))    
        print(f'Figure saved in {njoin(FIGS_DIR, fig_file)}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])