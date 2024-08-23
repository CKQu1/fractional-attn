import torch
from datasets import load_dataset
from transformers import BertTokenizer

# # Load the IWSLT14 dataset
# #dataset = load_dataset('iwslt14', 'de-en')

# src_language, trg_language = 'de', 'en'
# dataset = load_dataset("ted_talks_iwslt", language_pair=(src_language, trg_language), year="2014", split="train")

# # Initialize the tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['translation']['de'], examples['translation']['en'], padding="max_length", truncation=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # Convert to PyTorch datasets
# from torch.utils.data import DataLoader

# train_dataset = tokenized_datasets['train']
# valid_dataset = tokenized_datasets['validation']
# test_dataset = tokenized_datasets['test']

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# valid_dataloader = DataLoader(valid_dataset, batch_size=32)
# test_dataloader = DataLoader(test_dataset, batch_size=32)

# # Check the tokenized dataset
# print(tokenized_datasets['train'][0])


# ----------------------------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

max_length = 256
dataset = load_dataset("cfilt/iitb-english-hindi")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

article = dataset['validation'][2]['translation']['en']
inputs = tokenizer(article, return_tensors="pt")
 
translated_tokens = model.generate(
     **inputs,  max_length=256
 )
tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def preprocess_function(examples):
  inputs = [ex["en"] for ex in examples["translation"]]
  targets = [ex["hi"] for ex in examples["translation"]]
 
  model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
  labels = tokenizer(targets,max_length=max_length, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
 
  return model_inputs


tokenized_datasets_validation = dataset['validation'].map(
	preprocess_function,
	batched= True,
	remove_columns=dataset["validation"].column_names,
	batch_size = 2
)

tokenized_datasets_test = dataset['test'].map(
	preprocess_function,
	batched= True,
	remove_columns=dataset["test"].column_names,
	batch_size = 2)
