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

class FNSAttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the FNSMultiHeadAttention module.

    """

    def __init__(
        self,
        alpha,
        a,
        bandwidth,
        sphere_radius,
        hidden_size,
        attention_head_size,
        dropout,
        bias=True,
        is_cross_attention=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.alpha, self.bandwidth = alpha, bandwidth
        self.a = a
        self.sphere_radius = sphere_radius
        self.mask_val = math.pi * sphere_radius

        self.is_cross_attention = is_cross_attention

    def forward(self, x, encoder_output_states=None, attention_mask=None):
        if encoder_output_states is not None:
            assert (
                self.is_cross_attention
            ), "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = F.normalize(self.query(x), p=2, dim=-1)
            key = F.normalize(self.key(encoder_output_states), p=2, dim=-1)
            value = F.normalize(self.value(encoder_output_states), p=2, dim=-1)
        else:
            query = F.normalize(self.query(x), p=2, dim=-1)
            key = F.normalize(self.key(x), p=2, dim=-1)
            value = F.normalize(self.value(x), p=2, dim=-1)

        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a
        d_intrinsic = self.attention_head_size
        sphere_radius = self.sphere_radius
        mask_val = self.mask_val

        # geodesic distance on sphere
        eps = 1e-7  # for limiting the divergence from acos
        g_dist = (
            torch.acos(torch.clamp(query @ key.transpose(-1, -2), -1 + eps, 1 - eps))
            * sphere_radius
        )

        if attention_mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            if self.is_cross_attention:
                attention_mask = attention_mask[
                    :, :, : query.size(1), : key.size(1)
                ]  # Feels like a dirty fix...
            g_dist = g_dist.masked_fill_(attention_mask == 0, mask_val)

        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist / bandwidth**0.5) ** (-d_intrinsic - alpha)
        else:
            attn_score = torch.exp((-g_dist / bandwidth**0.5) ** (alpha / (alpha - 1)))

        if a == 0:
            # attn_score = attn_score.masked_fill_(attention_mask==0, -1e9) # Mask
            attention_probs = F.normalize(
                attn_score, p=1, dim=3
            )  # can do this as the attn weights are always positive
        else:
            D_inv_row = torch.diag_embed(
                attn_score.sum(-1) ** (-a)
            )  # inverse of degree matrix of attn_score
            D_inv_col = torch.diag_embed(
                attn_score.sum(-2) ** (-a)
            )  # inverse of degree matrix of attn_score
            K_tilde = D_inv_row @ attn_score @ D_inv_col
            # K_tilde = K_tilde.masked_fill_(attention_mask==0, -1e9) # Mask
            attention_probs = F.normalize(
                K_tilde, p=1, dim=3
            )  # can do this as the attn weights are always positive
        attention_probs = self.attn_dropout(attention_probs)

        # Calculate the attention output
        attention_output = attention_probs @ value

        return (attention_output, attention_probs)


class FNSMultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        # Whether it is cross attention
        self.is_cross_attention = is_cross_attention

        self.alpha = config["alpha"]
        self.bandwidth = config["bandwidth"]
        self.a = config["a"]
        self.sphere_radius = config["sphere_radius"]

        for _ in range(self.num_attention_heads):
            head = FNSAttentionHead(
                self.alpha,
                self.a,
                self.bandwidth,
                self.sphere_radius,
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias,
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


class FasterFNSMultiHeadAttention(nn.Module):
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
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.alpha = config["alpha"]
        self.bandwidth = config["bandwidth"]
        self.a = config["a"]
        self.sphere_radius = config["sphere_radius"]
        self.mask_val = config["mask_val"]

    def forward(
        self,
        x,
        attention_mask=None,
        output_attentions=False,
        encoder_output_states=None,
    ):

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
        num_attention_heads, attention_head_size = (
            self.num_attention_heads,
            self.attention_head_size,
        )

        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a
        sphere_radius = self.sphere_radius
        mask_val = self.mask_val
        d_intrinsic = attention_head_size

        query = F.normalize(
            query.view(
                batch_size,
                src_sequence_length,
                num_attention_heads,
                attention_head_size,
            ).transpose(1, 2),
            p=2,
            dim=-1,
        )
        key = F.normalize(
            key.view(
                batch_size,
                trg_sequence_length,
                num_attention_heads,
                attention_head_size,
            ).transpose(1, 2),
            p=2,
            dim=-1,
        )
        value = value.view(
            batch_size, trg_sequence_length, num_attention_heads, attention_head_size
        ).transpose(1, 2)

        # geodesic distance on sphere
        eps = 1e-7  # for limiting the divergence from acos
        g_dist = (
            torch.acos(torch.clamp(query @ key.transpose(-1, -2), -1 + eps, 1 - eps))
            * sphere_radius
        )

        if attention_mask is not None:
            if self.is_cross_attention:
                attention_mask = attention_mask[
                    :, :, :src_sequence_length, :trg_sequence_length
                ]  # Feels like a dirty fix...
            ##### begin{debug} #####
            # print(f'g_dist shape: {g_dist.shape}')
            # print(f'attention_mask shape: {attention_mask.shape}')
            ##### end{debug} #####
            g_dist = g_dist.masked_fill_(attention_mask == 0, mask_val)

        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist / bandwidth**0.5) ** (-d_intrinsic - alpha)
        else:
            attn_score = torch.exp((-g_dist / bandwidth**0.5) ** (alpha / (alpha - 1)))

        if a == 0:
            # attn_score = attn_score.masked_fill_(attention_mask.expand(-1,self.num_attention_heads,-1,-1)==0, -1e9) # Mask
            attention_probs = F.normalize(
                attn_score, p=1, dim=3
            )  # can do this as the attn weights are always positive
        else:
            D_inv_row = torch.diag_embed(
                attn_score.sum(-1) ** (-a)
            )  # inverse of degree matrix of attn_score
            D_inv_col = torch.diag_embed(
                attn_score.sum(-2) ** (-a)
            )  # inverse of degree matrix of attn_score
            K_tilde = D_inv_row @ attn_score @ D_inv_col
            # K_tilde = K_tilde.masked_fill_(attention_mask.expand(-1,self.num_attention_heads,-1,-1)==0, -1e9) # Mask
            attention_probs = F.normalize(
                K_tilde, p=1, dim=3
            )  # can do this as the attn weights are always positive
        attention_probs = self.attn_dropout(attention_probs)

        # ##### CHANGES HERE #####
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

class FNSEncoderBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterFNSMultiHeadAttention(config)
        else:
            self.attention = FNSMultiHeadAttention(config)
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

class FNSEncoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
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
            block = FNSEncoderBlock(config)
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


class ClassificationModel(nn.Module):
    """ FNSFormer classificaton model.
    """

    def __init__(self, config):
        """
        FNSEncoder + classifier head.
        """
        self.padding_idx = config["src_pad_token_id"]
        self.encoder = FNSEncoder(config)
        self.classifier = nn.Linear(
            in_features=config["hidden_size"], 
            out_features=config["num_labels"])

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Find last non-padded index (CHECK)
        B, L, H = x.shape
        last_nonpadded_idx = torch.arange(B) * L + torch.sum(x != self.padding_idx, dim=1, dtype=torch.int)-1 # B
        # Calculate the encoder's output
        x, all_attentions = self.encoder(
            x, attention_mask=attention_mask, output_attentions=output_attentions
        )
        x = self.classifier(x)
        # Get last non-padded output for each batch (no pool), B x H 
        x = x.view(B*L, H)[last_nonpadded_idx].view(B, H)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)

class DualClassifierHead(nn.Module):
    """Classifier head for dual encoding problem."""
    # See classifier_head_dual in LRA > lra_benchmarks > models > layers > common_layers.
    def __init__(self, config):
        self.pooling_mode = config["pooling_mode"]
        self.interaction = config["interaction"]
        self.linear1 = nn.Linear(in_features=config["hidden_size"] * 2, out_features=config["mlp_dim1"])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=config["mlp_dim1"], out_features=config["mlp_dim2"])
        self.linear3 = nn.Linear(in_features=config["mlp_dim2"], out_features=config["num_classes"])
        
    def forward(self, encoded1, encoded2):
        # Pooling across sequence length; B x L x H -> B x H
        if self.pooling_mode == 'MEAN':
            encoded1 = torch.mean(encoded1, dim=1)
            encoded2 = torch.mean(encoded2, dim=1)
        elif self.pooling_mode == 'SUM':
            encoded1 = torch.sum(encoded1, dim=1)
            encoded2 = torch.sum(encoded2, dim=1)
        elif self.pooling_mode == 'FLATTEN':
            encoded1 = encoded1.reshape((encoded1.shape[0], -1))
            encoded2 = encoded2.reshape((encoded2.shape[0], -1))
        elif self.pooling_mode == 'CLS':
            encoded1 = encoded1[:, 0, :]
            encoded2 = encoded2[:, 0, :]
        else:
            raise NotImplementedError('Pooling not supported yet.')
        if self.interaction == 'NLI':
            # NLI interaction style
            encoded = torch.cat([encoded1, encoded2, encoded1 * encoded2,
                                       encoded1 - encoded2], dim=1)
        else:
            encoded = torch.cat([encoded1, encoded2], dim=1)
        encoded = self.linear1(encoded)
        encoded = self.relu(encoded)
        encoded = self.linear2(encoded)
        encoded = self.relu(encoded)
        encoded = self.linear3(encoded)
        return encoded


class RetrievalModel(nn.Module): # CHECK
    """FNSFormer retrieval model for matching (dual encoding) tasks."""
    def __init__(self, config):
        """
        FNSEncoder + mean pooling + MLP.
        """
        self.encoder = FNSEncoder(config)
        self.classifier = DualClassifierHead(config)

    def forward(self, x1, x2, attention_mask1=None, attention_mask2=None, output_attentions=False):
        # Calculate encoder output for both inputs
        x1, all_attentions1 = self.encoder(
            x1, attention_mask=attention_mask1, output_attentions=output_attentions
        ) # B x L x H
        x2, all_attentions2 = self.encoder(
            x2, attention_mask=attention_mask2, output_attentions=output_attentions
        ) # B x L x H
        x = self.classifier(x1, x2) # B x H
        if not output_attentions:
            return (x, None)
        else:
            return (x, (all_attentions1, all_attentions2))