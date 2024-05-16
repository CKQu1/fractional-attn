# Import libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset

from constants import *
from mutils import njoin


def preprocess_function(examples, tokenizer_src, src_language, tokenizer_trg, trg_language, max_length):
    inputs = [example[src_language] for example in examples["translation"]]
    targets = [example[trg_language] for example in examples["translation"]]
    model_inputs = tokenizer_src(inputs, text_target=targets, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer_trg(targets, max_length=max_length, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs


def prepare_data(tokenizer_src, tokenizer_trg, batch_size=4, num_workers=2, test_fraction=0.2, max_length=512):
    # Load dataset; ignore validation set (tst2013) and use test set only (tst2014)
    src_language, trg_language = 'de', 'en'
    dataset = load_dataset("ted_talks_iwslt", language_pair=(src_language, trg_language), year="2014")
    dataset = dataset["train"].train_test_split(test_size=test_fraction, shuffle=True)
    trainset, testset = dataset['train'], dataset['test']
    # Preprocess datasets
    tokenized_trainset = trainset.map(lambda examples: preprocess_function(examples, tokenizer_src, src_language, tokenizer_trg, trg_language, max_length), batched=True)
    tokenized_testset = testset.map(lambda examples: preprocess_function(examples, tokenizer_src, src_language, tokenizer_trg, trg_language, max_length), batched=True)
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(tokenized_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(tokenized_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader