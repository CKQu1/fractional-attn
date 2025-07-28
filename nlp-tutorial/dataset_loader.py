from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

dataset_name = 'emotion'
max_len = 512

dataset = load_dataset(dataset_name)
train_dataset = dataset['train']
test_dataset = dataset['test']

# Load a tokenizer (Example: BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define a tokenization function with max length
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
# Set format for PyTorch
tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create a DataLoader for batching
batch_size = 16
train_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized_test_dataset, batch_size=batch_size, shuffle=True)