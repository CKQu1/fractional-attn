import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from ast import literal_eval
from os.path import isdir, join

sys.path.append(os.getcwd())
from path_setup import droot

def preprocess_df(df):
    col_names = ["eval_accuracy", "eval_f1_score"]  # can add more col_names
    N = df.shape[0]
    for col_name in col_names:
        assert col_name in df.columns
        for i in range(N):
            dct = literal_eval(df.loc[i,col_name])
            keys = list(dct.keys())
            assert len(keys) == 1
            df.loc[i,col_name] = dct[keys[0]]
    return df

def plot_single(model_dir):

    run_perf = pd.read_csv(join(model_dir, "run_performance.csv"))
    run_perf = preprocess_df(run_perf)
    metric_names = ["loss", "eval_loss", "eval_accuracy", "eval_f1_score"]

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, constrained_layout=True)
    axs = axs.flat

    for idx, metric_name in enumerate(metric_names):
        axis = axs[idx]
        axis.plot(run_perf.loc[:,'step'], run_perf.loc[:,metric_name], '.')
        axis.set_title(metric_name)

    fig_dir = join(droot, "figures_ms", "model_performance")
    if not isdir(fig_dir): os.makedirs(fig_dir)

    uuid_ = model_dir.split("/")[-1]
    if len(uuid_) == 0:
        uuid_ = model_dir.split("/")[-2]
    plt.savefig(join(fig_dir, f"{uuid_}.pdf"))
    print(f"Plot done for model {uuid_} from {model_dir}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])