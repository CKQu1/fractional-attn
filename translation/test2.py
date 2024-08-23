from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# tokenizer_src = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')  # German tokenizer
# tokenizer_trg = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')  # English tokenizer
tokenizer_src = AutoTokenizer.from_pretrained('dbmdz/german-gpt2')  # German tokenizer
tokenizer_trg = AutoTokenizer.from_pretrained('openai-community/gpt2')  # English tokenizer

def preprocess_function(examples, tokenizer_src, src_language, trg_language, max_length):
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
    dataset = dataset.train_test_split(test_size=test_fraction, shuffle=True)
    trainset, testset = dataset['train'], dataset['test']
    # Preprocess datasets
    tokenized_trainset = trainset.map(lambda examples: preprocess_function(examples, tokenizer_src, src_language, tokenizer_trg, trg_language, max_length), batched=True)
    tokenized_testset = testset.map(lambda examples: preprocess_function(examples, tokenizer_src, src_language, tokenizer_trg, trg_language, max_length), batched=True)
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(tokenized_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(tokenized_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader

dataset = load_dataset("ted_talks_iwslt", language_pair=("de", "en"), year="2014")

dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True)

testset = dataset["test"]

tokenized_testset = testset.map(lambda examples: preprocess_function(examples, tokenizer_src, "de", "en", 128), batched=True)

testloader = torch.utils.data.DataLoader(tokenized_testset, batch_size=2, shuffle=False, num_workers=1)

batch_size = 2
ix = torch.randint(len(testloader), (batch_size,))
x = torch.stack([data.dataset[i][0] for i in ix])
y = torch.tensor([data.dataset[i][1] for i in ix])

config = {
    "beta": 1,
    "bandwidth": 1,
    "sphere_radius": 1,
    "hidden_size": 1,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "num_attention_heads": 1,
    "intermediate_size": 4, # 4 * hidden_size
    "hidden_dropout_prob": 0,
    "encoder_dropout_prob": 0,
    "decoder_dropout_prob": 0,
    "attention_probs_dropout_prob": 0,
    "initializer_range": 0.1,
    "qkv_bias": True,
    "use_faster_attention": True,
    "src_vocab_size": tokenizer_src.vocab_size,
    "src_pad_token_id": tokenizer_src.pad_token_id,
    "trg_vocab_size": tokenizer_trg.vocab_size,
    "trg_pad_token_id": tokenizer_trg.pad_token_id,
    "max_length": 128,
}

from models.translation import DPForNMT
model = DPForNMT(config)

def greedy_decode(model, source, tokenizer_trg, max_len, device):
    bos_idx = tokenizer_trg.bos_token_id
    eos_idx = tokenizer_trg.eos_token_id

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(bos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build causal mask for target 
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int)
        decoder_mask = (decoder_mask == 0).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

source = torch.tensor(testloader.dataset["input_ids"][0])
source_mask = testloader.dataset["attention_mask"][0]
model.encoder(source, source_mask)




batch_size = 2
def get_batch(split):
    if split == 'train':
        data = trainloader
    else:
        data = testloader
    ix = torch.randint(len(data), (batch_size,))
    x = torch.tensor([testloader.dataset["input_ids"][i] for i in ix]) # (B, N) 
    y = torch.tensor([testloader.dataset["labels"][i] for i in ix]) # (B, N) 
    encoder_mask = torch.tensor([testloader.dataset["attention_mask"][i] for i in ix]) # (B, N) 
    encoder_mask = (encoder_mask.unsqueeze(-1)@encoder_mask.unsqueeze(1)).view(batch_size, 1, config["max_length"], config["max_length"]) # (B,1,N,N)
    decoder_mask = torch.stack([(torch.tensor(testloader.dataset["attention_mask"][i] != 0)).unsqueeze(0).int() & causal_mask(config["max_length"]) for i in ix]) # (B,1,N,N)
    return x, y    