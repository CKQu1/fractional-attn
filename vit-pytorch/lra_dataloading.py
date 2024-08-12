"""Adapted from https://github.com/lindermanlab/S5/blob/main/s5/dataloaders/lra.py"""

import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Tuple, Union

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
RAW_DATASET_DIR = '../.raw_datasets/lra_release/lra_release'

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, int, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int], Optional[int]], ReturnType]


# Example interface for making a loader.
def custom_loader(cache_dir: str,
				  train_bs: int = 50,
				  eval_bs: int = 50,
				  seed: int = 42) -> ReturnType:
	...


def make_data_loader(dset,
					 dobj,
					 seed: int,
					 batch_size: int=128,
					 shuffle: bool=True,
					 drop_last: bool=True,
					 collate_fn: callable=None):
	"""

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:return:
	"""

	# Create a generator for seeding random number draws.
	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	if dobj is not None:
		assert collate_fn is None
		collate_fn = dobj._collate_fn

	# Generate the dataloaders.
	return torch.utils.data.DataLoader(dataset=dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle,
									   drop_last=drop_last, generator=rng)


def create_lra_imdb_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										   train_bs: int = 50,
										   eval_bs: int = 50,
										   seed: int = 42) -> ReturnType:
	"""

	:param cache_dir:		(str):		Not currently used.
	:param train_bs:		(int):		Batch size for training.
 	:param eval_bs:			(int):		Batch size for evaluation.
	:param seed:			(int)		Seed for shuffling data.
	:return:
	"""
	print("[*] Generating LRA-text (IMDB) Classification Dataset")
	from lra_dataloaders.lra import IMDB
	name = 'imdb'

	dataset_obj = IMDB('imdb', )
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trainloader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
	#testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)
	valloader = None

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 135  # We should probably stop this from being hard-coded.
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	VOCAB_SIZE = dataset_obj.n_tokens

	return dataset_obj, trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE, VOCAB_SIZE


def create_lra_listops_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										   	  train_bs: int = 50,
										   	  eval_bs: int = 50,
											  seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-listops Classification Dataset")
	from lra_dataloaders.lra import ListOps
	name = 'listops'
	dir_name = f'{RAW_DATASET_DIR}/listops-1000'

	dataset_obj = ListOps(name, data_dir=dir_name)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trainloader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
	#valloader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	valloader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
	#testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 20
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	VOCAB_SIZE = dataset_obj.n_tokens

	return dataset_obj, trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE, VOCAB_SIZE


def create_lra_path32_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											 train_bs: int = 50,
											 eval_bs: int = 50,
											 seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-Pathfinder32 Classification Dataset")
	from lra_dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 32
	dir_name = f'{RAW_DATASET_DIR}/pathfinder{resolution}'

	#kwargs = {'tokenize': True}  # Tokenize into vocabulary of 256 values.
	kwargs = {'tokenize': False, 'sequential': False}

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
	#val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
	#tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

	N_CLASSES = dataset_obj.d_output
	#SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	SEQ_LENGTH = dataset_obj.dataset_train[0][0].shape[0]
	IN_DIM = dataset_obj.d_input
	#TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	VOCAB_SIZE = dataset_obj.n_tokens

	return dataset_obj, trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE, VOCAB_SIZE


def create_lra_pathx_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											train_bs: int = 50,
											eval_bs: int = 50,	
											seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-PathX Classification Dataset")
	from lra_dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 128
	dir_name = f'{RAW_DATASET_DIR}/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
	#val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
	#tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

	N_CLASSES = dataset_obj.d_output
	#SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	SEQ_LENGTH = dataset_obj.dataset_train[0][0].shape[0]
	IN_DIM = dataset_obj.d_input
	#TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	VOCAB_SIZE = dataset_obj.n_tokens

	return dataset_obj, trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE, VOCAB_SIZE


def create_lra_image_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											seed: int = 42,
											train_bs: int=128,
           									eval_bs: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating LRA-listops Classification Dataset")
	from lra_dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': True,  # LRA uses a grayscale CIFAR image.
		'tokenize': True, # Tokenize into vocabulary of 256 values.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)  # TODO - double check what the dir here does.
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=train_bs)
	#val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=eval_bs)
	#tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=eval_bs)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	VOCAB_SIZE = 256 # GRAYSCALE, no padding

	return dataset_obj, trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE, VOCAB_SIZE


def create_lra_aan_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										  train_bs: int = 50,
										  eval_bs: int = 50,
										  seed: int = 42, ) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-AAN Classification Dataset")
	from lra_dataloaders.lra import AAN
	name = 'aan'

	dir_name = f'{RAW_DATASET_DIR}/tsv_data'

	kwargs = {
		'n_workers': 1,  # Multiple workers seems to break AAN.
	}

	dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

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

	return dataset_obj, trn_loader, val_loader, tst_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE, VOCAB_SIZE


Datasets = {
	# LRA.
	"imdb-classification": create_lra_imdb_classification_dataset,
	"listops-classification": create_lra_listops_classification_dataset,
	"aan-classification": create_lra_aan_classification_dataset,
	"lra-cifar-classification": create_lra_image_classification_dataset,
	"pathfinder-classification": create_lra_path32_classification_dataset,
	"pathx-classification": create_lra_pathx_classification_dataset,
}
