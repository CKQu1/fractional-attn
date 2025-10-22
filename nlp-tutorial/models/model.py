import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

from .model_utils import PositionWiseFeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model = config['d_model']
        self.n_heads = n_heads = config['n_heads']
        self.p_drop = p_drop = config['p_drop']
        self.d_ff = d_ff = config['d_ff']

        self.is_resnet_scale = config['is_resnet_scale']
        if self.is_resnet_scale:
            self.n_layers = config['n_layers']

        self.model_name = config['model_name']
        if self.model_name[-8:] == 'dpformer':
            from .att_dp import MultiHeadAttention, OPMultiHeadAttention
            if not config['is_op']:
                self.mha = MultiHeadAttention(config, is_return_dist)
            else:
                self.mha = OPMultiHeadAttention(config, is_return_dist)
        elif self.model_name[-11:] == 'rdfnsformer':
            from .att_rdfns import FNSAttention, OPFNSAttention
            if not config['is_op']:
                self.mha = FNSAttention(config, is_return_dist)
            else:
                self.mha = OPFNSAttention(config, is_return_dist)

        elif self.model_name[-11:] == 'spfnsformer':
            from .att_spfns import FNSAttention, OPFNSAttention
            if not config['is_op']:
                self.mha = FNSAttention(config, is_return_dist)
            else:
                self.mha = OPFNSAttention(config, is_return_dist)

        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.is_return_dist = is_return_dist

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        # only normalize if is spherical
        if self.model_name[-11:] == 'spfnsformer':
            inputs = F.normalize(inputs, p=2, dim=-1)

        if not self.is_return_dist:
            attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        else:
            attn_outputs, attn_weights, g_dist = self.mha(inputs, inputs, inputs, attn_mask)

        attn_outputs = self.dropout1(attn_outputs)
        if self.is_resnet_scale:  
            attn_outputs = self.layernorm1(inputs + attn_outputs/(self.n_layers)**0.5)
        else:
            attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        if not self.is_return_dist:
            return ffn_outputs, attn_weights
        else:
            return ffn_outputs, attn_weights, g_dist

class Transformer(nn.Module):
    
    # def __init__(self, vocab_size, seq_len, d_model=512, n_layers=6, n_heads=8, p_drop=0.1, d_ff=2048, pad_id=0):
    #     super(Transformer, self).__init__()
    #     self.pad_id = pad_id
    #     self.sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model) # (seq_len+1, d_model)

    #     # layers
    #     self.embedding = nn.Embedding(vocab_size, d_model)
    #     self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
    #     self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    #     # layers to classify
    #     self.linear = nn.Linear(d_model, 2)
    #     self.softmax = nn.Softmax(dim=-1)

    def __init__(self, config, is_return_dist=False):
        super(Transformer, self).__init__()

        self.vocab_size = config['vocab_size']
        self.seq_len = config['seq_len']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.p_drop = config['p_drop']
        self.d_ff = config['d_ff']
        self.pad_id = config['pad_id']
        self.fix_embed = config['fix_embed']
        self.max_len = config['seq_len']
        self.num_classes = config['num_classes']
        if self.fix_embed:
            self.pretrained_model_name = config['pretrained_model_name']

        self.is_return_dist = is_return_dist

        # embeddings
        if not self.fix_embed:
            self.sinusoid_table = self.get_sinusoid_table(self.seq_len+1, self.d_model) # (seq_len+1, d_model)
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
            # self.pos_embedding = nn.Embedding(self.seq_len+1, self.d_model)
            self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        elif (self.fix_embed and config['pretrained_model_name'] == 'glove'):
            self.sinusoid_table = self.get_sinusoid_table(self.seq_len+1, self.d_model) # (seq_len+1, d_model)
            self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)                                             
            # change method to executing this in main.py
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        elif (self.fix_embed and config['pretrained_model_name'] in\
             ['distilbert-base-uncased', 'albert-base-v2']):
            self.pos_embedding = nn.Embedding(self.seq_len+1, self.d_model)
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        elif (self.fix_embed and config['pretrained_model_name'] == 'gpt2'):
            self.pos_embedding = nn.Embedding(self.seq_len+1, self.d_model)
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)                  

        # layers        
        self.layers = nn.ModuleList([EncoderLayer(config, self.is_return_dist) for _ in range(self.n_layers)])

        # layers to classify
        self.linear = nn.Linear(self.d_model, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)                                     

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        if not self.fix_embed or (self.fix_embed and self.pretrained_model_name in\
             ['glove', 'distilbert-base-uncased', 'albert-base-v2', 'gpt2']):
            outputs = self.embedding(inputs) + self.pos_embedding(positions)
            # |outputs| : (batch_size, seq_len, d_model)
        else:
            #outputs = self.embeddings(input_ids=inputs, position_ids=positions)
            outputs = self.embeddings(input_ids=inputs)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        attention_weights = []
        if not self.is_return_dist:
            for layer in self.layers:
                outputs, attn_weights = layer(outputs, attn_pad_mask)
                # |outputs| : (batch_size, seq_len, d_model)
                # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
                attention_weights.append(attn_weights)
        else:
            g_dists = []
            for layer in self.layers:        
                outputs, attn_weights, g_dist = layer(outputs, attn_pad_mask)
                # |outputs| : (batch_size, seq_len, d_model)
                # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
                attention_weights.append(attn_weights)
                g_dists.append(g_dist)                

        outputs, _ = torch.max(outputs, dim=1)
        # |outputs| : (batch_size, d_model)
        outputs = self.softmax(self.linear(outputs))
        # |outputs| : (batch_size, 2)

        if not self.is_return_dist:
            return outputs, attention_weights
        else:
            return outputs, attention_weights, g_dists

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)