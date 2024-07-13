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
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )

class DPAttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the DPMultiHeadAttention module.

    """

    def __init__(
        self,
        hidden_size,
        attention_head_size,
        dropout,
        bias=True,
        qk_share=False,
        is_cross_attention=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        self.use_key = not qk_share or is_cross_attention
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        if self.use_key:
            self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.is_cross_attention = is_cross_attention

        self.qk_share = qk_share

    def forward(self, x, encoder_output_states=None, attention_mask=None):
        if encoder_output_states is not None:
            assert (
                self.is_cross_attention
            ), "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.query(x)
            key = self.key(encoder_output_states)
            value = self.value(encoder_output_states)
        else:
            query = self.query(x)
            if self.use_key:
                key = self.key(x)
            value = self.value(x)

        if self.use_key:
            attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        else:
            attn_score = torch.matmul(query, query.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            if self.is_cross_attention:
                attention_mask = attention_mask[
                    :, :, : query.size(1), : key.size(1)
                ]  # Feels like a dirty fix...
            attn_score = attn_score.masked_fill_(attention_mask == 0, -1e9)

        attention_probs = F.softmax(attn_score, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # Calculate the attention output
        attention_output = attention_probs @ value

        return (attention_output, attention_probs)


class DPMultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        self.qk_share = config["qk_share"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        # Whether it is cross attention
        self.is_cross_attention = is_cross_attention

        for _ in range(self.num_attention_heads):
            head = DPAttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias,
                self.qk_share,
                self.is_cross_attention,
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(
        self,
        x,
        attention_mask=None,
        output_attentions=False,
        encoder_output_states=None,
    ):
        # Calculate the attention output for each attention head
        attention_outputs = [
            head(x, encoder_output_states, attention_mask) for head in self.heads
        ]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], dim=-1
        )
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], dim=1
            )
            return (attention_output, attention_probs)


class FasterDPMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention

        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        self.qk_share = config["qk_share"]
        # Create a linear layer to project the query, key, and value
        """
        if self.is_cross_attention:
            self.kv_projection = nn.Linear(
                self.hidden_size, self.all_head_size * 2, bias=self.qkv_bias
            )
            self.q_projection = nn.Linear(
                self.hidden_size, self.all_head_size, bias=self.qkv_bias
            )
        else:
            self.qkv_projection = nn.Linear(
                self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias
            )
        """
        self.use_key = not self.qk_share or self.is_cross_attention  # whether to use key proj

        self.q_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        if self.use_key:
            self.k_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        self.v_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)

        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(
        self,
        x,
        attention_mask=None,
        output_attentions=False
    ):

        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        #qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        #query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = self.q_projection(x)
        if self.use_key:
            key = self.k_projection(x)
        value = self.v_projection(x)

        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, src_sequence_length, _ = query.size()
        trg_sequence_length = key.size(1) if self.use_key else src_sequence_length
        num_attention_heads, attention_head_size = (
            self.num_attention_heads,
            self.attention_head_size,
        )

        query = query.view(batch_size, src_sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        if self.use_key:
            key = key.view(batch_size, trg_sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        value = value.view( batch_size, trg_sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)

        if self.use_key:
            attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        else:
            attn_score = torch.matmul(query, query.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            if self.is_cross_attention:
                attention_mask = attention_mask[
                    :, :, : query.size(1), : key.size(1)
                ]  # Feels like a dirty fix...
            attn_score = attn_score.masked_fill_(attention_mask == 0, -1e9)

        attention_probs = F.softmax(attn_score, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # print(f'attention_probs shape: {attention_probs.shape}')

        # Calculate the attention output
        attention_output = attention_probs @ value

        # ##### CHANGES HERE #####
        # print(f'attention_output shape: {attention_output.shape}')
        # print('\n')

        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, src_sequence_length, self.all_head_size)
        )
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

class DPEncoderBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterDPMultiHeadAttention(config)
        else:
            self.attention = DPMultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(
            x, attention_mask=attention_mask, output_attentions=output_attentions
        )
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

class DPEncoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Embeddings
        self.pooling_mode = config["pooling_mode"]
        self.token_embedding = nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["hidden_size"],
            padding_idx=config["padding_idx"],
        )
        if self.pooling_mode == 'CLS':
            self.cls_token = nn.Parameter(torch.randn(1,1,config["hidden_size"]))
            num_embeddings = config["max_length"] + 1
        else:
            num_embeddings = config["max_length"]
        self.positional_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=config["hidden_size"],
        )
        self.dropout = nn.Dropout(p=config["encoder_dropout_prob"])
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_encoder_layers"]):
            block = DPEncoderBlock(config)
            self.blocks.append(block)

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        token_embeddings = self.token_embedding(x) # B x L x H
        if self.pooling_mode == 'CLS':
            # Prepend the CLS token
            cls_token = self.cls_token.expand(x.shape[0], -1, -1) # B x 1 x H
            token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)
        position_ids = torch.arange(0, token_embeddings.shape[1]).to(x.device)
        position_embeddings = self.positional_embedding(position_ids)
        # Dropout
        x = self.dropout(position_embeddings + token_embeddings)
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(
                x, attention_mask=attention_mask, output_attentions=output_attentions
            )
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)