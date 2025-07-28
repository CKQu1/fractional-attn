from typing import Iterable, Union, List
from prenlp.data import IMDB

import torch
from torch.utils.data import TensorDataset

DATASETS_CLASSES = {'imdb': IMDB}

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

from datasets import load_dataset
from torch.utils.data import DataLoader
from constants import DROOT
from .mutils import njoin

def get_datasets(args, tokenizer):

    # Load the dataset
    if '-' not in args.dataset_name:
        dataset = load_dataset(args.dataset_name, cache_dir=njoin(DROOT, 'DATASETS'))
    else:
        dataset_name, subset_name = args.dataset_name.split('-')
        dataset = load_dataset(dataset_name, subset_name, cache_dir=njoin(DROOT, 'DATASETS'))
    train_dataset = dataset['train']
    if 'sst' not in args.dataset_name:
        test_dataset = dataset['test']
    else:
        test_dataset = dataset['validation']

    # Define a tokenization function with max length
    def tokenize_function(examples, text_key='text'):
        return tokenizer(examples[text_key], padding="max_length", truncation=True, max_length=args.max_len)

    # Tokenize the dataset
    text_key = 'text' if 'sst' not in args.dataset_name else 'sentence'
    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, text_key=text_key), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(x, text_key=text_key), batched=True)

    # Set format for PyTorch
    tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_train_dataset, tokenized_test_dataset

# Custom collate function to return only (input_ids, labels)
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return input_ids, labels

def datasets_create_examples(args, tokenized_train_dataset, tokenized_test_dataset):
    # Create a DataLoader for batching
    batch_size = args.train_bs
    train_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn)    
    return train_loader, test_loader