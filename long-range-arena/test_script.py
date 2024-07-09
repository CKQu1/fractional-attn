from dataloading import Datasets

create_dataset_fn = Datasets["pathfinder-classification"]

trainloader, valloader, testloader, n_classes, seq_len, in_dim, train_size = \
    create_dataset_fn(cache_dir='/cache_dir', seed=0, train_bs=2, eval_bs=2) 

# Create iterators for getting individual batches
train_iter = iter(trainloader)
val_iter = iter(valloader)
test_iter = iter(testloader)

batch = next(train_iter)
if len(batch) == 2:
    inputs, targets = batch
    aux_data = {}
elif len(batch) == 3:
    inputs, targets, aux_data = batch
# Grab lengths from aux if it is there.
lengths = aux_data.get('lengths', None)