from mutils import njoin
from constants import *

dataset_name = 'listops-classification'
cache_dir = njoin(DROOT, 'cache_dir')
seed = 42

train_batch_size = eval_batch_size = 2

from lra_dataloading import Datasets

# Get dataset creation function
create_dataset_fn = Datasets[dataset_name]

# Dataset dependent logic
if dataset_name in ["imdb-classification", "listops-classification", "aan-classification"]:
    padded = True
    if dataset_name in ["aan-classification"]:
        # Use retreival model for document matching
        retrieval = True
        print("Using retrieval model for document matching")
    else:
        retrieval = False
else:
    padded = False
    retrieval = False
    
# Create dataset...
dataset_obj, trainloader, valloader, testloader, num_classes, seq_len, in_dim, train_size, vocab_size = \
    create_dataset_fn(cache_dir=cache_dir, seed=seed, train_bs=train_batch_size, eval_bs=eval_batch_size)
eval_size = len(testloader.dataset)  

dataset, tokenizer, vocab = dataset_obj._load_from_cache(dataset_obj.cache_dir / dataset_obj._cache_dir_name)