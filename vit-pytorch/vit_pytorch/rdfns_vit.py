import math
import torch
from torch import nn
from torch.nn import functional as F

from vit_pytorch.model_utils import NewGELUActivation, PatchEmbeddings, Embeddings, MLP


class DMFNSAttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the DMFNSMultiHeadAttention module.

    """
    def __init__(self, config):
        super().__init__()
        # The attention head size is the hidden size divided by the number of attention heads
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.query = nn.Linear(self.hidden_size, self.attention_head_size, bias=self.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.attention_head_size, bias=self.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.attention_head_size, bias=self.qkv_bias)

        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])

        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']

        #self.sphere_radius = config['sphere_radius']

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)                

        batch_size, sequence_length, _ = query.size()
        num_attention_heads, attention_head_size = self.num_attention_heads, self.attention_head_size

        query = query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)            

        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a

        #sphere_radius = self.sphere_radius
        d_intrinsic = self.attention_head_size

        # Euclidean dist in R^d   
        g_dist = torch.cdist(query, key, p=2) 

        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))
        attn_score_shape = attn_score.shape

        # print(f'hidden_size = {self.hidden_size}, attention_head_size = {self.attention_head_size}')
        # print(f'sequence_length = {sequence_length}')
        # print(f'query shape: {query.shape}')
        # print(f'key shape: {key.shape}')
        # print(f'value shape: {value.shape}')  
        # print(f'g_dist shape: {g_dist.shape}') 
        # print(f'attn_score_shape: {attn_score_shape}') 
        # print(f'diag shape: {torch.diag_embed(attn_score.sum(-1)**(-a)).shape}')

        if a > 0:
            # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col sum
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)            
            attention_probs = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attention_probs = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive

        attention_probs = self.attn_dropout(attention_probs)

        # Calculate the attention output
        attention_output = attention_probs @ value

        return (attention_output, attention_probs)


class DMFNSMultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config['num_attention_heads']
        # # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]


        # Create a list of attention heads
        self.heads = nn.ModuleList([])

        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']

        #self.sphere_radius = config['sphere_radius']     

        for _ in range(self.num_attention_heads):
            head = DMFNSAttentionHead(
                config
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
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


class FasterDMFNSMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        self.qk_share = config["qk_share"]
        # Create a linear layer to project the query, key, and value
        if not self.qk_share:
            self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        else:
            self.qv_projection = nn.Linear(self.hidden_size, self.all_head_size * 2, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']
        #self.sphere_radius = config['sphere_radius']

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        if not self.qk_share:            
            # Split the projected query, key, and value into query, key, and value
            # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
            query, key, value = torch.chunk(self.qkv_projection(x), 3, dim=-1)
        else:            
            query, value = torch.chunk(self.qv_projection(x), 2, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        num_attention_heads, attention_head_size = self.num_attention_heads, self.attention_head_size

        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a

        #sphere_radius = self.sphere_radius
        d_intrinsic = attention_head_size

        query = query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        # print(f'query shape: {query.shape}')
        # print(f'key shape: {key.shape}')
        # print(f'value shape: {value.shape}')        

        # Euclidean dist in R^d
        g_dist = torch.cdist(query, key, p=2)        
        
        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))
        attn_score_shape = attn_score.shape
        if a > 0:
            # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col sum
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)

            attention_probs = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:                      
            attention_probs = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive
        attention_probs = self.attn_dropout(attention_probs)

        # Calculate the attention output
        attention_output = attention_probs @ value
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class DMFNSBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterDMFNSMultiHeadAttention(config)
        else:
            self.attention = DMFNSMultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    # Norm & Add (Pre-LayerNorm)
    # def forward(self, x, output_attentions=False):
    #     # Self-attention
    #     attention_output, attention_probs = \
    #         self.attention(self.layernorm_1(x), output_attentions=output_attentions)
    #     # Skip connection
    #     x = x + attention_output
    #     # Feed-forward network
    #     mlp_output = self.mlp(self.layernorm_2(x))
    #     # Skip connection
    #     x = x + mlp_output
    #     # Return the transformer block's output and the attention probabilities (optional)
    #     if not output_attentions:
    #         return (x, None)
    #     else:
    #         return (x, attention_probs)

    # Add & Norm (Post-LayerNorm)
    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(x, output_attentions=output_attentions)    
        # Skip connection + layernorm
        x = self.layernorm_1(x + attention_output)                
        # Feed-forward network
        mlp_output = self.mlp(x)     
        # Skip connection + layernorm
        x = self.layernorm_2(x + mlp_output)           
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)              


class DMFNSEncoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = DMFNSBlock(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class RDFNSViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = DMFNSEncoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)