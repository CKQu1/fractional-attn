import argparse
from os.path import isfile
from pathlib import Path
from dataloading import make_data_loader
from mutils import njoin
from constants import DROOT

"""
This is for test loading aan/pathfinder to debug
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='For quickly processing raw datasets.')  
    parser.add_argument('--name', default='pathfinder', type=str)
    args = parser.parse_args()

    name = args.name

    # load $PBS_JOBFS (NEED TO RUN lra_dl.sh first)
    if isfile(njoin(DROOT, 'jobfs_path.txt')):
        with open(njoin(DROOT, 'jobfs_path.txt')) as f:
            RAW_DATA_PATH = f.read().strip()

        print(f'RAW_DATA_PATH = {RAW_DATA_PATH} \n')
    else:
        print(njoin(DROOT, 'jobfs_path.txt') + ' does not exist, run lra_dl.sh first!')
        quit()

    if name == 'aan':
        from dataloaders.lra import AAN

        #dir_name = './.raw_datasets/lra_release/lra_release/tsv_data'
        #dir_name = njoin(DROOT, 'lra_release', 'lra_release', 'tsv_data')
        dir_name = njoin(RAW_DATA_PATH, 'lra_release', 'lra_release', 'tsv_data')

        kwargs = {
            'n_workers': 1,  # Multiple workers seems to break AAN.
        }

        #cache_dir = '/taiji1/chqu7424/fractional-attn/long-range-arena/.droot/cache_dir'
        cache_dir = njoin(DROOT, 'cache_dir')

        dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
        dataset_obj.cache_dir = Path(cache_dir) / name
        dataset_obj.setup()

        seed = 42
        train_bs = eval_bs = 32

        trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
        #val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
        val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
        #tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
        tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

        print('AAN loaded!')

    elif name == 'pathfinder':
        from dataloaders.lra import PathFinder
        
        resolution = 32
        #dir_name = f'{RAW_DATA_PATH}/lra_release/lra_release/pathfinder{resolution}'
        dir_name = njoin(RAW_DATA_PATH, 'lra_release', 'lra_release', f'pathfinder{resolution}')

        kwargs = {'tokenize': True}  # Tokenize into vocabulary of 256 values.
        cache_dir = njoin(DROOT, 'cache_dir')

        dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution, **kwargs)
        dataset_obj.cache_dir = Path(cache_dir) / name
        dataset_obj.setup()

        seed = 42
        train_bs = eval_bs = 32

        trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
        #val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
        val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
        #tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
        tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)


        print('PATHFINDER loaded!')

    else:
        print('Dataset does not exist!')

# -----------------------------------------------------------------

# from datasets import load_dataset

# data_dir = '/taiji1/chqu7424/fractional-attn/long-range-arena/\
# .raw_datasets/lra_release/lra_release/tsv_data'

# dataset = load_dataset(
#     "csv",
#     data_files={
#         "train": str(data_dir + "new_aan_pairs.train.tsv"),
#         "val": str(data_dir + "new_aan_pairs.eval.tsv"),
#         "test": str(data_dir + "new_aan_pairs.test.tsv"),
#     },
#     delimiter="\t",
#     column_names=["label", "input1_id", "input2_id", "text1", "text2"],
#     keep_in_memory=True,
# )  # True)