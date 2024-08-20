import tensorflow as tf
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import numpy as np
from os.path import isdir, isfile
from os import makedirs

from constants import *
from mutils import *

# Specify the path to your .tfevents file
#event_file = '.droot/full-models/lra-listops/opfns-listops-alpha=1.2-eps=1-a=0/seed=0/test.tensorboard/events.out.tfevents.1723821971.hpc219.225203.0'
def load_tfevents(event_file):
    global event_acc, events

    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Initialize lists to store the data
    tags = event_acc.Tags()['scalars']
    data = {tag: [] for tag in tags}

    # Extract scalar events
    for tag in tags:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append((event.wall_time, event.step, event.value))

    # Convert to DataFrame
    df_list = []
    for tag in tags:
        df = pd.DataFrame(data[tag], columns=['wall_time', 'step', 'value'])
        df['tag'] = tag
        df_list.append(df)

    # Concatenate all dataframes
    final_df = pd.concat(df_list)

    tags = final_df.loc[:,'tag'].unique()
    DCT = {}
    for tag in tags:
        DCT[tag] = final_df[final_df['tag']==tag]

    return DCT

def get_event_file(model_dir):
    seed_path_dct = {}
    dirnames = os.listdir(model_dir)
    for dirname in dirnames:
        seed_path = njoin(model_dir, dirname)
        tensorboard_path = njoin(seed_path, 'test.tensorboard')
        if isdir(tensorboard_path):
            for f in os.listdir(tensorboard_path):
                if 'tfevents' in f:
                    seed_path_dct[dirname] = [seed_path, f]        

    return seed_path_dct

def fig_metrics(root, fns_type='spopfns', iter_size=100, tag='accu',
                excluded_models = ['sink'],
                display=False, cbar_separate=True):
    global seed_path_dct, seed_paths
    global event_file, DCT
    global tag_values, steps
    global model_dirs, model_types_plotted
    global task_idx, task_root

    iter_size = int(iter_size)
    display = str2bool(display)
    cbar_separate = str2bool(cbar_separate)

    task_roots = []
    for dirname in os.listdir(root):
        if 'lra-' in dirname:
            task_roots.append(njoin(root, dirname))
    nrows, ncols = 1, len(task_roots)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'

    model_types_plotted = []
    for task_idx, task_root in enumerate(task_roots):
        task_name = task_root.split('/')[-1].split('-')[1]
        model_dirs = []
        for model_dir in os.listdir(task_root):
            if 'fns' not in model_dir:
                model_dirs.append(njoin(task_root, model_dir))    
            else:
                if fns_type in model_dir:
                    model_dirs.append(njoin(task_root, model_dir))
        for model_dir in model_dirs:            
            seed_path_dct = get_event_file(model_dir)
            seed_paths = list(seed_path_dct.keys())
            model_dirname = model_dir.split('/')[-1]
            model_setup = {}
            for model_hyp in model_dirname.split('-')[2:]:
                k, v = model_hyp.split('=')
                model_setup[k] = v
            model_type = model_dirname.split('-')[0]
            model_label = NAMES_DICT[model_type]
            if 'fns' in model_type:
                alpha = float(model_setup['alpha'])
                c_hyp = HYP_CMAP(HYP_CNORM(alpha))
                # print(f'alpha = {alpha}')  # delete
            #else:
            elif model_type not in excluded_models:
                c_hyp = OTHER_COLORS_DICT[model_type]
            ii = 0
            if len(seed_paths) > 0:
                for seed_path in seed_paths:
                    event_file = njoin(seed_path_dct[seed_path][0], 'test.tensorboard', seed_path_dct[seed_path][1])  
                    try:
                        DCT = load_tfevents(event_file)
                        assert tag in DCT.keys()
                        if ii==0:
                            tag_values = DCT[tag].loc[::iter_size,'value']
                            steps = DCT[tag].loc[::iter_size,'step']
                            #steps = DCT[tag].loc[::iter_size,'wall_time']
                        else:
                            tag_values += DCT[tag].loc[::iter_size,'value']
                        ii += 1
                    except:
                        continue
                if ii > 0:
                    tag_values = tag_values/ii
                    # print('tag_values')
                    # print(tag_values)
                    axs[task_idx].plot(steps, tag_values, c=c_hyp, linestyle=LINESTYLE_DICT[model_type])      

                    if model_type not in model_types_plotted:
                        model_types_plotted.append(model_type)                      

        axs[task_idx].set_title(task_name)
        axs[task_idx].grid()

    # legend
    # for alpha in alphas:
    #     axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
    #                 label=rf'$\alpha$ = {alpha}')    

    for model_type in model_types_plotted:   
        if 'fns' in model_type:
            color = 'k'
        else:
            color = OTHER_COLORS_DICT[model_type]
        axs[0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], 
                    label=NAMES_DICT[model_type])

    ncol_legend = 2 if len(model_types_plotted) == 3 else 1
    if len(model_types_plotted) >= 2:
        axs[0].legend(loc='best', ncol=ncol_legend, frameon=False)           
        # axs[0].legend(bbox_to_anchor=(0.85, 1.35),
        #             loc='best', ncol=ncol_legend, frameon=False)                        

    # Add shared x and y labels
    # fig.text(0.5, 0.01, 'Epochs', fontsize='medium', ha='center')
    # fig.text(0, 0.5, NAMES_DICT[metrics[0]], fontsize='medium', va='center', rotation='vertical')       
    fig.supxlabel('Epochs', fontsize='medium')
    fig.supylabel(tag.upper(), fontsize='medium')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the right label                 

    from constants import FIGS_DIR
    SAVE_DIR = FIGS_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)
    if display:
        plt.show()
    else:
        os.makedirs(SAVE_DIR, exist_ok=True)
        fig_dir = njoin(SAVE_DIR, '-'.join(model_types_plotted) + f'-{tag}.pdf')
        print(f'Figure saved as {fig_dir}')
        plt.savefig(fig_dir)

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