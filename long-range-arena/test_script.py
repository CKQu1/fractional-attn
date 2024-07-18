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

# -------------------------------------------
from pathlib import Path
cache_dir = DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

from dataloaders.lra import PathFinder
name = 'pathfinder'
resolution = 32
dir_name = f'./.raw_datasets/lra_release/lra_release/pathfinder{resolution}'

dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
dataset_obj.cache_dir = Path(cache_dir) / name