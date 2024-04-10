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
                                 max_seq_len: int) -> List[InputFeatures]:
    pad_token_id = tokenizer.pad_token_id
    
    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = tokens[:max_seq_len]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        label_id = label_dict.get(example.label)
        
        feature = InputFeatures(input_ids, label_id)
        features.append(feature)

    return features

def create_examples(args,
                    tokenizer,
                    mode: str = 'train') -> Iterable[Union[List[InputExample], dict]]:
    if mode == 'train':
        dataset = DATASETS_CLASSES[args.dataset]()[0]
    elif mode == 'test':
        dataset = DATASETS_CLASSES[args.dataset]()[1]

    examples = []
    for text, label in dataset:
        example = InputExample(text, label)
        examples.append(example)
    
    labels = sorted(list(set([example.label for example in examples])))
    label_dict = {label: i for i, label in enumerate(labels)}
    # print('[{}]\tLabel dictionary:\t{}'.format(mode, label_dict))

    features = convert_examples_to_features(examples, label_dict, tokenizer, args.max_seq_len)
    
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_label_ids)
    
    if args.divider > 1:
        import random
        N = len(dataset)
        idxs = random.sample(list(range(N)), int(N/args.divider))
        dataset = torch.utils.data.Subset(dataset, idxs)        

    return dataset

# ---------------------------------------- Newly added (2024/03/28) ----------------------------------------

from datasets import load_dataset

# simple wrapper for load_dataset
def get_dataset(dataset_name, cache_dir):
    if dataset_name in ['imdb', 'emotion', 'rotten_tomatoes']:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    elif dataset_name == "hyperpartisan_news_detection":
        dataset = load_dataset(dataset_name, "byarticle", cache_dir=cache_dir)
    elif dataset_name == "newsgroup":
        dataset = load_dataset(dataset_name, '18828_alt.atheism', cache_dir=cache_dir)   
    elif dataset_name == "cola":
        dataset = load_dataset('glue', 'cola', cache_dir=cache_dir) 
    else:
        raise NameError(f"{dataset_name} not applicable!")
    return dataset

# get dataset feature names
def get_dataset_cols(dataset):
    return list(dataset.column_names.values())[0]

# remove and rename key names in dataset
def process_dataset_cols(dataset):
    column_names = get_dataset_cols(dataset)
    if 'hyperpartisan' in column_names:
        dataset = dataset.rename_column("hyperpartisan", "label")
    if 'sentence' in column_names:
        dataset = dataset.rename_column('sentence', 'text')
    
    # for removing unused features i.e. text
    column_names = list(dataset.column_names.values())[0]
    remove_columns = list(set(column_names) - {'input_ids', 'attention_mask', 'label'})
    if len(remove_columns) > 0:
        dataset = dataset.map(remove_columns=remove_columns)
    
    return dataset

def preprocess_logits_for_metrics(logits):
    preds = logits.argmax(dim=-1)
    return preds

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    acc = metric_acc.compute(predictions=preds, references=labels)        
    f1_score = metric_f1.compute(predictions=preds, references=labels)  
    #precision = metric_prcn.compute(predictions=preds, references=labels)        
    #recall = metric_recall.compute(predictions=preds, references=labels)                
    #return {"accuracy":acc,"f1_score":f1_score,"precision":precision,"recall":recall}
    return {"accuracy":acc,"f1_score":f1_score}