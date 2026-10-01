import os
from pathlib import Path
from typing import Iterable, Union, List
from prenlp.data import IMDB

import torch
from torch.utils.data import TensorDataset

DATASETS_CLASSES = {'imdb': IMDB}

HF_DATASET_CONFIGS = {
    'ag_news': {
        'path': 'ag_news',
        'train_split': 'train',
        'test_split': 'test',
        'text_key': 'text',
        'label_key': 'label',
        'num_classes': 4,
        'tokenizer_name': 'bert-base-uncased',
    },
    'glue-sst2': {
        'path': 'glue',
        'name': 'sst2',
        'train_split': 'train',
        'test_split': 'validation',
        'text_key': 'sentence',
        'label_key': 'label',
        'num_classes': 2,
        'tokenizer_name': 'bert-base-uncased',
    },
}

class InputExample:
    """A single training/test example for text classification.
    """
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label

class InputFeatures:
    """A single set of features of data.
    """
    def __init__(self, input_ids: List[int], label_id: int):
        self.input_ids = input_ids
        self.label_id = label_id

def convert_examples_to_features(examples: List[InputExample],
                                 label_dict: dict,
                                 tokenizer,
                                 max_len: int) -> List[InputFeatures]:
    pad_token_id = tokenizer.pad_token_id
    
    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = tokens[:max_len]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        label_id = label_dict.get(example.label)
        
        feature = InputFeatures(input_ids, label_id)
        features.append(feature)

    return features

def create_examples(args,
                    tokenizer,
                    mode: str = 'train') -> Iterable[Union[List[InputExample], dict]]:
    if mode == 'train':
        #dataset = DATASETS_CLASSES[args.dataset]()[0]
        dataset = DATASETS_CLASSES[args.dataset_name]()[0]
    elif mode == 'test':
        #dataset = DATASETS_CLASSES[args.dataset]()[1]
        dataset = DATASETS_CLASSES[args.dataset_name]()[1]

    examples = []
    for text, label in dataset:
        example = InputExample(text, label)
        examples.append(example)
    
    labels = sorted(list(set([example.label for example in examples])))
    label_dict = {label: i for i, label in enumerate(labels)}
    # print('[{}]\tLabel dictionary:\t{}'.format(mode, label_dict))

    features = convert_examples_to_features(examples, label_dict, tokenizer, args.max_len)
    
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_label_ids)

    return dataset

# ---------- GLoVe Embeddings ----------
 
#from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
#from torchtext.datasets import IMDB
from torch.nn.utils.rnn import pad_sequence     

def tokenize_and_numericalize(text, vocab, tokenizer):
    tokens = tokenizer(text)
    return [vocab[token] if token in vocab else 0 for token in tokens]

# Tokenize, numericalize, and pad sequences
def process_data(args, iterator, vocab, tokenizer):
    data, labels = [], []
    #for label, text in iterator:
    for text, label in iterator:
        token_ids = tokenize_and_numericalize(text, vocab, tokenizer)
        data.append(torch.tensor(token_ids[:args.max_len]))
        labels.append(1 if label == 'pos' else 0)  # Convert labels to numeric
    return pad_sequence(data, batch_first=True), torch.tensor(labels)    

def glove_create_examples(args, glove_dim, tokenizer, mode: str = 'train'):

    if mode == 'train':
        #dataset = DATASETS_CLASSES[args.dataset]()[0]
        dataset = DATASETS_CLASSES[args.dataset_name]()[0]
    elif mode == 'test':
        #dataset = DATASETS_CLASSES[args.dataset]()[1]
        dataset = DATASETS_CLASSES[args.dataset_name]()[1]    

    #tokenizer = get_tokenizer("basic_english")
    #glove = GloVe(name='6B', dim=100)  # Example for 100d embeddings    
    #glove = GloVe(name='6B', dim=args.hidden)
    glove = GloVe(name='6B', dim=glove_dim)
    VOCAB = glove.stoi  # String-to-Index mapping

    #x_train, y_train = process_data(train_iter, vocab, tokenizer)
    all_input_ids, all_label_ids = process_data(args, dataset, VOCAB, tokenizer)

    dataset = TensorDataset(all_input_ids, all_label_ids)

    return dataset

# ---------- datasets API ----------

from datasets import DownloadConfig, load_dataset
from datasets import config as datasets_config
from torch.utils.data import DataLoader
from constants import DROOT
from .mutils import njoin

def get_hf_dataset_config(dataset_name: str) -> dict:
    if dataset_name in HF_DATASET_CONFIGS:
        return HF_DATASET_CONFIGS[dataset_name].copy()

    config = {
        'path': dataset_name,
        'train_split': 'train',
        'test_split': 'test',
        'text_key': 'text',
        'label_key': 'label',
        'tokenizer_name': 'bert-base-uncased',
    }
    if '-' in dataset_name:
        dataset_path, subset_name = dataset_name.split('-', maxsplit=1)
        config['path'] = dataset_path
        config['name'] = subset_name
    if 'sst' in dataset_name:
        config['test_split'] = 'validation'
        config['text_key'] = 'sentence'

    return config

def get_num_classes(dataset, label_key: str, configured_num_classes=None):
    if configured_num_classes is not None:
        return configured_num_classes

    label_feature = dataset.features.get(label_key)
    if hasattr(label_feature, 'num_classes'):
        return label_feature.num_classes

    return len(set(dataset[label_key]))

def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in ('yes', 'true', 't', 'y', '1')
    return bool(value)

def set_huggingface_cache_dir(cache_dir: str, local_files_only=False) -> str:
    cache_dir = os.path.abspath(cache_dir)
    cache_path = Path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(njoin(cache_dir, 'modules'), exist_ok=True)
    os.makedirs(njoin(cache_dir, 'downloads'), exist_ok=True)

    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['HF_MODULES_CACHE'] = njoin(cache_dir, 'modules')
    if as_bool(local_files_only):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        datasets_config.HF_DATASETS_OFFLINE = True

    datasets_config.HF_DATASETS_CACHE = cache_path
    datasets_config.HF_MODULES_CACHE = cache_path / 'modules'
    datasets_config.DOWNLOADED_DATASETS_PATH = cache_path / 'downloads'

    return cache_dir

def get_datasets(args, tokenizer):

    # Load the dataset
    dataset_config = get_hf_dataset_config(args.dataset_name)
    local_files_only = as_bool(getattr(args, 'local_files_only', False))
    cache_dir = set_huggingface_cache_dir(
        args.cache_dir if getattr(args, 'cache_dir', None) else njoin(DROOT, 'hf_cache_dir'),
        local_files_only=local_files_only,
    )
    dataset_kwargs = {'cache_dir': cache_dir}
    if local_files_only:
        dataset_kwargs['download_config'] = DownloadConfig(local_files_only=True)
    if 'name' in dataset_config:
        dataset = load_dataset(dataset_config['path'], dataset_config['name'], **dataset_kwargs)
    else:
        dataset = load_dataset(dataset_config['path'], **dataset_kwargs)
    train_dataset = dataset[dataset_config['train_split']]
    test_dataset = dataset[dataset_config['test_split']]
    text_key = dataset_config['text_key']
    label_key = dataset_config['label_key']
    num_classes = get_num_classes(train_dataset, label_key, dataset_config.get('num_classes'))

    # Define a tokenization function with max length
    def tokenize_function(examples):
        return tokenizer(examples[text_key], padding="max_length", truncation=True, max_length=args.max_len)

    # Tokenize the dataset
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", label_key])
    tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", label_key])

    return tokenized_train_dataset, tokenized_test_dataset, num_classes, label_key

# Custom collate function to return only (input_ids, labels)
def collate_fn(batch, label_key='label'):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    label_values = [item[label_key] for item in batch]
    if torch.is_tensor(label_values[0]):
        labels = torch.stack(label_values).long()
    else:
        labels = torch.tensor(label_values, dtype=torch.long)
    return input_ids, labels

def datasets_create_examples(args, tokenized_train_dataset, tokenized_test_dataset, label_key='label'):
    # Create a DataLoader for batching
    batch_size = args.train_bs
    def batch_collate_fn(batch):
        return collate_fn(batch, label_key=label_key)

    train_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=batch_collate_fn)
    test_loader = DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=batch_collate_fn)
    return train_loader, test_loader

# ---------- dataset utility ----------
 
def count_trailing_zeros(tensor):
    # Reverse the tensor and find the first non-zero element
    reversed_tensor = torch.flip(tensor, dims=[0])
    nonzero_indices = torch.nonzero(reversed_tensor, as_tuple=True)[0]
    
    if len(nonzero_indices) == 0:  # All elements are zero
        return len(tensor)
    else:  # Count zeros from the end
        return nonzero_indices[0].item()    
