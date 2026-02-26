import torch
from torch import nn
from .interfaces import Module
from .embedding import Embedding
from .encoder import EncoderLayer
from .decoder import DecoderLayer


class Transformer(Module):
    def __init__(self, config):
        super().__init__()

        self.src_pad_index = config['src_pad_index']
        self.trg_pad_index = config['trg_pad_index']

        # Manually seed to keep embeddings consistent across loads
        # torch.manual_seed(config['seed'])

        # Embeddings, pass in pad indices to prevent <pad> from contributing to gradient
        self.src_embedding = Embedding(config['d_model'],
                                       config['src_vocab_len'],
                                       config['src_pad_index'],
                                       dropout_rate=config['dropout_rate'])
        self.trg_embedding = Embedding(config['d_model'],
                                       config['trg_vocab_len'],
                                       config['trg_pad_index'],
                                       dropout_rate=config['dropout_rate'])

        # Encoder
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(config)
             for _ in range(config['num_layers'])]
        )

        # Decoder
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(config)
             for _ in range(config['num_layers'])]
        )

        # Final layer to project embedding to target vocab word probability distribution
        self.linear = nn.Linear(config['d_model'], config['trg_vocab_len'])

        # Move to GPU if possible
        self.to(self.device)

        # Re-seed afterward to allow shuffled data
        # torch.seed()

    def forward(self, source, target):
        # Encoder stack
        enc_out = self.src_embedding(source)
        for layer in self.encoder_stack:
            enc_out = layer(enc_out)

        # Decoder stack
        dec_out = self.trg_embedding(target)
        for layer in self.decoder_stack:
            dec_out = layer(dec_out, enc_out)

        # Final linear layer to get word probabilities
        # DO NOT apply softmax here, as CrossEntropyLoss already does softmax!!!
        return self.linear(dec_out)
