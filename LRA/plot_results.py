import tensorflow as tf
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
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

def fig_metrics(root, iter_size=100, tag='accu'):
    global seed_path_dct, seed_paths
    global event_file, DCT
    global tag_values
    iter_size = int(iter_size)

    task_roots = []
    for dirname in os.listdir(root):
        if 'lra-' in dirname:
            task_roots.append(njoin(root, dirname))
    nrows, ncols = 1, len(task_roots)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'

    for task_idx, task_root in enumerate(task_roots):
        task_name = task_root.split('/')[-1].split('-')[1]
        model_dirs = [njoin(task_root, model_dir) for model_dir in os.listdir(task_root)]
        for model_dir in model_dirs:            
            seed_path_dct = get_event_file(model_dir)
            seed_paths = list(seed_path_dct.keys())
            model_name = model_dir.split('/')[-1]
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
                        else:
                            tag_values += DCT[tag].loc[::iter_size,'value']
                        ii += 1
                    except:
                        continue
                if ii > 0:
                    tag_values = tag_values/ii
                    axs[task_idx].plot(steps, tag_values, label=model_name+f'avg={ii}')                    

        axs[task_idx].set_title(task_name)
    axs[0].legend()

    SAVE_PATH = njoin(DROOT, 'figs_dir')
    os.makedirs(SAVE_PATH, exist_ok=True)
    fig_dir = njoin(SAVE_PATH, f'model-{tag}.pdf')
    print(f'Figure saved as {fig_dir}')
    plt.savefig(fig_dir)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])    