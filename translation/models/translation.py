import math
import torch
from torch import nn


import math
import torch
from torch import nn
from torch.nn import functional as F

# https://github.com/tintn/vision-transformer-from-scratch/blob/main/vit.py
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True, is_cross_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
        
        self.is_cross_attention = is_cross_attention
    
    def forward(self, x, encoder_output_states=None, attention_mask=None):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        if encoder_output_states is not None:
            assert self.is_cross_attention, "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.query(x)
            key = self.key(encoder_output_states)
            value = self.value(encoder_output_states)            
        else:
            query = self.query(x)
            key = self.key(x)
            value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_mask = attention_mask[:,:,:query.size(1),:key.size(1)] # Feels like a dirty fix...
            attention_scores.masked_fill_(attention_mask == 0, -1e9)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias,
                self.is_cross_attention
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, attention_mask=None, output_attentions=False, encoder_output_states=None):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x, encoder_output_states, attention_mask) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        if self.is_cross_attention:
            self.kv_projection = nn.Linear(self.hidden_size, self.all_head_size * 2, bias=self.qkv_bias)
            self.q_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        else:
            self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, attention_mask=None, output_attentions=False, encoder_output_states=None):
        # Project the query, key, and value
        if encoder_output_states is not None:
            assert hasattr(
                self, "q_projection"
            ), "If class is used as cross attention, the weights `q_projection` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_projection(x)
            kv = self.kv_projection(encoder_output_states)
            key, value = torch.chunk(kv, 2, dim=-1)
        else:
            # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
            qkv = self.qkv_projection(x)
            # Split the projected query, key, and value into query, key, and value
            # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
            query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, src_sequence_length, _ = query.size()
        trg_sequence_length = key.size(1)
        query = query.view(batch_size, src_sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, trg_sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, trg_sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_mask = attention_mask[:,:,:src_sequence_length,:trg_sequence_length] # Feels like a dirty fix...
            attention_scores.masked_fill_(attention_mask == 0, -1e9)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, src_sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(x, attention_mask=attention_mask, output_attentions=output_attentions)
        # Skip connection
        x = self.layernorm_1(x + attention_output)
        # Feed-forward network
        mlp_output = self.mlp(x)
        # Skip connection
        x = self.layernorm_2(x + mlp_output)
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        

class DecoderBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.self_attention = FasterMultiHeadAttention(config)
            self.cross_attention = FasterMultiHeadAttention(config, is_cross_attention=True)
        else:
            self.self_attention = MultiHeadAttention(config)
            self.cross_attention = MultiHeadAttention(config, is_cross_attention=True)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_3 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, encoder_output_states, src_mask=None, trg_mask=None, output_attentions=False):
        # Self-attention
        attention_output, self_attention_probs = \
            self.self_attention(x, attention_mask=trg_mask, output_attentions=output_attentions)
        # Skip connection
        x = self.layernorm_1(x + attention_output)
        # Cross-attention
        attention_output, cross_attention_probs = \
            self.cross_attention(x, attention_mask=src_mask, output_attentions=output_attentions, encoder_output_states=encoder_output_states)
        # Skip connection
        x = self.layernorm_2(x + attention_output)
        # Feed-forward network
        mlp_output = self.mlp(x)
        # Skip connection
        x = self.layernorm_3(x + mlp_output)
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None, None)
        else:
            return (x, self_attention_probs, cross_attention_probs)
        

class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.padding_idx = config["src_pad_token_id"]
        # Embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=config["src_vocab_size"],
            embedding_dim=config["hidden_size"],
            padding_idx=config["src_pad_token_id"],
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=config["max_length"],
            embedding_dim=config["hidden_size"],
        )
        self.dropout = nn.Dropout(p=config["encoder_dropout_prob"])
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_encoder_layers"]):
            block = EncoderBlock(config)
            self.blocks.append(block)

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = torch.arange(0, x.shape[-1]).to(x.device)
        position_embeddings = self.positional_embedding(position_ids)
        token_embeddings = self.token_embedding(x)
        # Dropout 
        x = self.dropout(position_embeddings + token_embeddings)
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, attention_mask=attention_mask, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)    
    
class Decoder(nn.Module):
    """
    The transformer decoder module.
    """

    def __init__(self, config, bias=True):
        super().__init__()
        self.padding_idx = config["trg_pad_token_id"]
        # Embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=config["trg_vocab_size"],
            embedding_dim=config["hidden_size"],
            padding_idx=config["trg_pad_token_id"],
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=config["max_length"],
            embedding_dim=config["hidden_size"],
        )
        self.dropout = nn.Dropout(p=config["decoder_dropout_prob"])
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_decoder_layers"]):
            block = DecoderBlock(config)
            self.blocks.append(block)
        # Tie output linear weights to input embedding matrix
        self.fc = nn.Linear(config["hidden_size"], config["trg_vocab_size"], bias=bias)
        self.fc.weight = self.token_embedding.weight 
        
    def forward(self, x, embedding_output_states, src_mask=None, trg_mask=None, output_attentions=False):
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = torch.arange(0, x.shape[-1]).to(x.device)
        position_embeddings = self.positional_embedding(position_ids)
        token_embeddings = self.token_embedding(x)
        # Dropout 
        x = self.dropout(position_embeddings + token_embeddings)
        # Calculate the transformer block's output for each block
        all_self_attentions = []
        all_cross_attentions = []
        for block in self.blocks:
            if output_attentions:
                x, self_attention_probs, cross_attention_probs = block(x, embedding_output_states, src_mask=src_mask, trg_mask=trg_mask, output_attentions=output_attentions)
                all_self_attentions.append(self_attention_probs)
                all_cross_attentions.append(cross_attention_probs)
            else: 
                x, _, _ = block(x, embedding_output_states, src_mask=src_mask, trg_mask=trg_mask, output_attentions=output_attentions)              
        # Linear layer
        x = self.fc(x)
        # Softmax
        x = nn.Softmax(dim=-1)(x)
        # Return logits and the attention probabilities (optional)
        if not output_attentions:
            return (x, None, None)
        else:
            return (x, all_self_attentions, all_cross_attentions)

class DPForNMT(nn.Module):
    """
    The seq2seq model for neural machine translation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create the transformer decoder module
        self.decoder = Decoder(config)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, src, trg, encoder_mask, decoder_mask, output_attentions=False):
        # Calculate the encoder's output
        encoder_output, encoder_self_attentions = self.encoder(src, attention_mask = encoder_mask, output_attentions=output_attentions)
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            # Calculate the decoder's output
            decoder_output, _, _ = self.decoder(trg, encoder_output, src_mask=encoder_mask, trg_mask=decoder_mask, output_attentions=output_attentions)
            return (decoder_output, None, None, None)
        else:
            # Calculate the decoder's output
            decoder_output, decoder_self_attentions, decoder_cross_attentions = self.decoder(trg, encoder_output, src_mask=encoder_mask, trg_mask=decoder_mask, output_attentions=output_attentions)
            return (decoder_output, encoder_self_attentions, decoder_self_attentions, decoder_cross_attentions)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)