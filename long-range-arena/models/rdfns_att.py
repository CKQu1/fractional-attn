import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

from models.model_utils import NewGELUActivation, MLP

torch.autograd.set_detect_anomaly(True)
class RDFNSMultiHeadAttention(nn.Module):
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
        self.qk_share = config['qk_share']

        self.use_key = not self.qk_share or self.is_cross_attention
        self.is_op = config["is_op"]        

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
        # self.q_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        # if self.use_key:
        #     self.k_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        # self.v_projection = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        if self.is_op:
            if not self.qk_share:
                self.WK = orthogonal(nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias))
            if self.num_attention_heads > 1:
                self.WQ = orthogonal(nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias))
        else:
            self.WQ = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
            if not self.qk_share:
                self.WK = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)

        self.WV = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.alpha = config["alpha"]
        self.bandwidth = config["bandwidth"]
        self.a = config["a"]
        if self.alpha < 2:
            self.d_intrinsic = config["d_intrinsic"]
        self.mask_val = config["mask_val"]

        self.is_rescale_dist = config['is_rescale_dist']
        if self.is_rescale_dist:
            if self.alpha >= 2:
                self.dist_scale = (self.attention_head_size)**0.5
            else:
                #self.dist_scale = (self.attention_head_size)**(1/self.alpha)
                #self.dist_scale = self.attention_head_size**0.5 / (self.attention_head_size**(1/self.attention_head_size) - 1)
                self.dist_scale = self.attention_head_size**0.5 / (2**(1/self.attention_head_size) - 1) 

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
        # query = self.q_projection(x)
        # if self.use_key:
        #     key = self.k_projection(x)     

        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a
        mask_val = self.mask_val
        if alpha < 2:
            d_intrinsic = self.d_intrinsic

        if not self.is_op or self.num_attention_heads > 1:
            query = self.WQ(x)
        else:
            query = x

        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, src_sequence_length, _ = query.size()        
        num_attention_heads, attention_head_size = self.num_attention_heads, self.attention_head_size

        ##### begin{debug} #####
        # print(f'query shape: {query.shape}')
        # print(f'({batch_size}, {src_sequence_length}, {num_attention_heads}, {attention_head_size})')
        ##### end{debug} #####

        # query = query.view(
        #             batch_size,
        #             src_sequence_length,
        #             num_attention_heads,
        #             attention_head_size,
        #         ).transpose(1, 2)
        # if self.use_key:
        #     key = key.view(
        #             batch_size,
        #             trg_sequence_length,
        #             num_attention_heads,
        #             attention_head_size,
        #         ).transpose(1, 2)

        query = query.view(batch_size, src_sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        # geodesic distance on R^d
        if not self.qk_share:            
            key = self.WK(x)
            trg_sequence_length = key.size(1)
            key = key.view(batch_size, trg_sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)            
            g_dist = torch.cdist(query, key, p=2)  # Euclidean dist in R^d             
        else:                      
            trg_sequence_length = src_sequence_length                         
            g_dist = torch.cdist(query, query, p=2)
        if self.is_rescale_dist:
            g_dist = g_dist / self.dist_scale

        value = self.WV(x)
        value = value.view(
            batch_size, trg_sequence_length, num_attention_heads, attention_head_size
        ).transpose(1, 2)        

        # if attention_mask is not None:
        #     g_dist = g_dist.masked_fill_(attention_mask == 0, mask_val)

        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist / bandwidth**0.5) ** (-d_intrinsic - alpha)
        else:            
            attn_score = torch.exp(-(g_dist / bandwidth**0.5) ** (alpha / (alpha - 1)))
        if attention_mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            if self.is_cross_attention:
                attention_mask = attention_mask[
                    :, :, : src_sequence_length, : trg_sequence_length
                ]  # Feels like a dirty fix...
            attn_score = attn_score.masked_fill(attention_mask == 0, mask_val)

        if a == 0:
            # attn_score = attn_score.masked_fill_(attention_mask.expand(-1,self.num_attention_heads,-1,-1)==0, -1e9) # Mask
            # can do this as the attn weights are always positive
            attention_probs = F.normalize(attn_score, p=1, dim=3)  
        else:
            """
            D_inv_row = torch.diag_embed(
                attn_score.sum(-1) ** (-a)
            )  # inverse of degree matrix of attn_score
            D_inv_col = torch.diag_embed(
                attn_score.sum(-2) ** (-a)
            )  # inverse of degree matrix of attn_score
            K_tilde = D_inv_row @ attn_score @ D_inv_col
            # K_tilde = K_tilde.masked_fill_(attention_mask.expand(-1,self.num_attention_heads,-1,-1)==0, -1e9) # Mask
            """
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col sum                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2) 

            attention_probs = F.normalize(K_tilde, p=1, dim=3)  # can do this as the attn weights are always positive
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


class RDFNSEncoderBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = RDFNSMultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])
        self.is_preln = config['is_preln']

    # Self-attention
    def forward(self, x, attention_mask=None, output_attentions=False):      
        # Norm & Add (Pre-LayerNorm)
        if self.is_preln:
            attention_output, attention_probs = self.attention(
                self.layernorm_1(x), attention_mask=attention_mask, output_attentions=output_attentions
            )
            # Skip connection
            x = x + attention_output
            # Feed-forward network
            mlp_output = self.mlp(self.layernorm_2(x))
            # Skip connection
            x = x + mlp_output        
        # Add & Norm (Post-LayerNorm)
        else:
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


class RDFNSEncoder(nn.Module):
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
            block = RDFNSEncoderBlock(config)
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