"""
This is for test loading aan to debug
"""
from dataloaders.lra import AAN
from pathlib import Path
from dataloading import make_data_loader

name = 'aan'

dir_name = './.raw_datasets/lra_release/lra_release/tsv_data'

kwargs = {
    'n_workers': 1,  # Multiple workers seems to break AAN.
}

cache_dir = '/taiji1/chqu7424/fractional-attn/long-range-arena/.droot/cache_dir'

dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
dataset_obj.cache_dir = Path(cache_dir) / name
dataset_obj.setup()

seed = 42
train_bs = 32

trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
#val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
#tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

N_CLASSES = dataset_obj.d_output
SEQ_LENGTH = dataset_obj.l_max
IN_DIM = len(dataset_obj.vocab)
TRAIN_SIZE = len(dataset_obj.dataset_train)
VOCAB_SIZE = dataset_obj.n_tokens

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