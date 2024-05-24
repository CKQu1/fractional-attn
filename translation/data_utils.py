# Import libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset

from constants import *
from mutils import njoin

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


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