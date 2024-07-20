# 1. -------------------------------------------

# from dataloading import Datasets

# #create_dataset_fn = Datasets["imdb-classification"]
# create_dataset_fn = Datasets["pathfinder-classification"]

# trainloader, valloader, testloader, n_classes, seq_len, in_dim, train_size = \
#     create_dataset_fn(cache_dir='long-range-arena/cache_dir/', seed=0, train_bs=2, eval_bs=2) 

# # # Create iterators for getting individual batches
# # train_iter = iter(trainloader)
# # val_iter = iter(valloader)
# # test_iter = iter(testloader)

# batch = next(iter(trainloader))
# if len(batch) == 2:
#     inputs, targets = batch
#     aux_data = {}
# elif len(batch) == 3:
#     inputs, targets, aux_data = batch
# # Grab lengths from aux if it is there.
# lengths = aux_data.get('lengths', None)

# 2. -------------------------------------------

import torch
from pathlib import Path
cache_dir = DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

from dataloaders.lra import PathFinder
name = 'pathfinder'
resolution = 32
dir_name = f'./.raw_datasets/lra_release/lra_release/pathfinder{resolution}'

kwargs = {
    'tokenize': True # Tokenize into vocabulary of 256 values.
}

dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution, **kwargs)
dataset_obj.cache_dir = Path(cache_dir) / name

# ---- dataloader.lra -----

from dataloaders.lra import PathFinderDataset

# under setup()
torch.multiprocessing.set_sharing_strategy("file_system")
dataset = PathFinderDataset(dataset_obj.data_dir, transform=dataset_obj.default_transforms())
len_dataset = len(dataset)
val_len = int(dataset_obj.val_split * len_dataset)
test_len = int(dataset_obj.test_split * len_dataset)
train_len = len_dataset - val_len - test_len
(
    dataset_obj.dataset_train,
    dataset_obj.dataset_val,
    dataset_obj.dataset_test,
) = torch.utils.data.random_split(
    dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(dataset_obj.seed),
)

from dataloading import make_data_loader

seed = 42
train_bs = 8
eval_bs = 8
trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
#val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
#tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

N_CLASSES = dataset_obj.d_output
#SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
SEQ_LENGH = dataset_obj.dataset_train[0][0].shape[0]
IN_DIM = dataset_obj.d_input
#TRAIN_SIZE = len(dataset_obj.dataset_train)
VOCAB_SIZE = dataset_obj.n_tokens

# 3. -------------------------------------------

# import torch
# from pathlib import Path
# cache_dir = DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

# from dataloaders.lra import PathFinder
# name = 'pathfinder'
# resolution = 32
# dir_name = f'./.raw_datasets/lra_release/lra_release/pathfinder{resolution}'

# kwargs = {'tokenize': True}  # Tokenize into vocabulary of 256 values.

# dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution, **kwargs)
# dataset_obj.cache_dir = Path(cache_dir) / name
# dataset_obj.setup()

# from dataloading import make_data_loader

# seed = 42
# train_bs = 8
# eval_bs = 8
# trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
# #val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
# val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
# #tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
# tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

# N_CLASSES = dataset_obj.d_output
# #SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
# SEQ_LENGH = dataset_obj.dataset_train[0][0].shape[0]
# IN_DIM = dataset_obj.d_input
# #TRAIN_SIZE = len(dataset_obj.dataset_train)
# VOCAB_SIZE = dataset_obj.n_tokens

# 4. -------------------------------------------

# from constants import DROOT
# from mutils import njoin
# from dataloading import Datasets

# dataset_name = 'pathfinder-classification'
# create_dataset_fn = Datasets[dataset_name]

# trainloader, valloader, testloader, num_classes, seq_len, in_dim, train_size, vocab_size = \
#     create_dataset_fn(cache_dir=njoin(DROOT, 'cache_dir'), seed=42, train_bs=2, eval_bs=2)