import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from ast import literal_eval
from os.path import isdir, isfile, join

sys.path.append(os.getcwd())
from path_setup import droot

def preprocess_df(df):
    col_names = ["eval_accuracy", "eval_f1_score"]  # can add more col_names
    N = df.shape[0]
    for col_name in col_names:
        assert col_name in df.columns
        for i in df.index:
            dct = df.loc[i,col_name]
            if isinstance(dct, str):
                dct = literal_eval(dct)
            keys = list(dct.keys())
            assert len(keys) == 1
            df.loc[i,col_name] = dct[keys[0]]
    return df

def extract_json(model_dir):
    global data, df_perf, sub_dirs, counter, epoch

    import json
    sub_dirs = [join(model_dir, sub_dir) for sub_dir in sorted(next(os.walk(model_dir))[1]) \
                 if "checkpoint-" in sub_dir]
    assert len(sub_dirs) > 0, "There are no checkpoints, please wait till training is more complete!"
    
    metrics_dict = {}
    for metric in ["epoch", "loss", "eval_loss", "eval_accuracy", "eval_f1_score"]:
        metrics_dict[metric] = []
    for idx, sub_dir in enumerate(sub_dirs):
        f = open(join(sub_dir, "trainer_state.json"))
        data = json.load(f)
        epoch = data['log_history'][-1]['epoch']
        metrics_dict['epoch'].append(epoch)
        counter = 0
        epoch_data = []
        for i in range(len(data['log_history'])):
            if data['log_history'][i]['epoch'] == epoch:
                counter += 1
            if counter == 1:
                metrics_dict['loss'].append(data['log_history'][i]['loss'])
            if counter == 2:
                metrics_dict['eval_loss'].append(data['log_history'][i]['eval_loss'])
                metrics_dict['eval_accuracy'].append(data['log_history'][i]['eval_accuracy']['accuracy'])
                metrics_dict['eval_f1_score'].append(data['log_history'][i]['eval_f1_score']['f1'])

        df_perf = pd.DataFrame(metrics_dict)

    return df_perf

def plot_single(model_dir, display=False):

    fpath = join(model_dir, "run_performance.csv")
    if not isfile(fpath):
        df_perf = extract_json(model_dir)
    else:
        df_perf = pd.read_csv(fpath, index_col=None)
        df_perf = df_perf.dropna()
        df_perf = preprocess_df(df_perf)

    metric_names = ["loss", "eval_loss", "eval_accuracy", "eval_f1_score"]
    titles = ["Loss", "Eval loss", "Eval acc", "Eval f1"]

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, constrained_layout=True)
    axs = axs.flat

    for idx, metric_name in enumerate(metric_names):
        axis = axs[idx]
        #axis.plot(df_perf.loc[:,'step'], df_perf.loc[:,metric_name], '.')
        axis.plot(df_perf.loc[:,'epoch'], df_perf.loc[:,metric_name], '-o')
        axis.set_title(titles[idx])

    axs[-2].set_xlabel("Epoch"); axs[-1].set_xlabel("Epoch")

    fig_dir = join(droot, "figures_ms", "model_performance")
    if not isdir(fig_dir): os.makedirs(fig_dir)

    uuid_ = model_dir.split("/")[-1]
    if len(uuid_) == 0:
        uuid_ = model_dir.split("/")[-2]
    if display:
        plt.show()
    else:
        plt.savefig(join(fig_dir, f"{uuid_}.pdf"))
        print(f"Plot done for model {uuid_} from {model_dir}")

def plot_multiple(model_root_dir, display=False):
    global df_perf, model_dirs

    metric_names = ["loss", "eval_loss", "eval_accuracy", "eval_f1_score"]
    titles = ["Loss", "Eval loss", "Eval acc", "Eval f1"]

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, constrained_layout=True)
    axs = axs.flat

    model_root_dirs = [join(model_root_dir, sub_dir) for sub_dir in next(os.walk(model_root_dir))[1] \
                       if "job" not in sub_dir]
    for model_root_dir in model_root_dirs:
        # remove empty dirs
        model_dirs = [join(model_root_dir, sub_dir) for sub_dir in next(os.walk(model_root_dir))[1] \
                       if "model=0" in sub_dir and \
                       not not os.listdir(join(model_root_dir, sub_dir))]  # can adjust to include more trained models here
        model_dirs = sorted(model_dirs)

        for model_dir in model_dirs:
            fpath = join(model_dir, "run_performance.csv")
            if not isfile(fpath):
                df_perf = extract_json(model_dir)
            else:
                df_perf = pd.read_csv(fpath, index_col=None)
                df_perf = df_perf.dropna()
                df_perf = preprocess_df(df_perf)
            # sort by epoch
            df_perf = df_perf.sort_values(by=['epoch'])

            if "gamma=" in  model_dir:
                gamma = model_dir.split("gamma=")[-1]
                if len(gamma) == 0:
                    gamma = model_dir.split("gamma=")[-2]
                model_type = rf"$\gamma$ = {gamma}"
            else:
                model_type = "Diffuser"
            for idx, metric_name in enumerate(metric_names):
                axis = axs[idx]
                #axis.plot(df_perf.loc[:,'step'], df_perf.loc[:,metric_name], '.')
                axis.plot(df_perf.loc[:,'epoch'], df_perf.loc[:,metric_name], '-o', label=model_type)
                axis.set_title(titles[idx])

    # extra settings
    axs[0].legend()
    axs[-2].set_xlabel("Epoch"); axs[-1].set_xlabel("Epoch")            

    dataset_name = "rotten_tomato"
    fig_dir = join(droot, "figures_ms", "model_performance")
    if not isdir(fig_dir): os.makedirs(fig_dir)    
    if display:
        plt.show()
    else:
        plt.savefig(join(fig_dir, f"dataset={dataset_name}.pdf"))
        print(f"Plot done for {model_root_dir}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])