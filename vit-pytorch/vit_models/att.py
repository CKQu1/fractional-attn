import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal


class MultiHeadAttention(nn.Module):
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
        self.is_op = config["is_op"]
        
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

    def forward(self, x, output_attentions=False):
        num_attention_heads, attention_head_size = self.num_attention_heads, self.attention_head_size

        if not self.is_op or self.num_attention_heads > 1:
            query = self.WQ(x)
        else:
            query = x
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()    
        query = query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)

        if not self.qk_share:
            key = self.WK(x)
            key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
            # Calculate the attention scores
            # softmax(Q*K.T/sqrt(head_size))*V
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
        else:
            attention_scores = torch.matmul(query, query.transpose(-1, -2))

        value = self.WV(x)
        value = value.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)

        attention_scores = attention_scores / math.sqrt(attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
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


class RDFNSMultiHeadAttention(nn.Module):
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
        self.is_op = config["is_op"]
        # Create a linear layer to project the query, key, and value
        # if not self.qk_share:
        #     self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        # else:
        #     self.qv_projection = nn.Linear(self.hidden_size, self.all_head_size * 2, bias=self.qkv_bias)
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

        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']
        #self.sphere_radius = config['sphere_radius']

        self.is_rescale_dist = config['is_rescale_dist']
        if self.is_rescale_dist:
            if self.alpha >= 2:
                self.dist_scale = (self.attention_head_size)**0.5
            else:
                #self.dist_scale = (self.attention_head_size)**(1/self.alpha)
                #self.dist_scale = self.attention_head_size**0.5 / (self.attention_head_size**(1/self.attention_head_size) - 1)

                #self.dist_scale = self.attention_head_size**0.5 / (2**(1/self.attention_head_size) - 1)        
                #self.dist_scale = self.attention_head_size**0.5 / (self.attention_head_size**(1/self.attention_head_size) - 1)
                #self.dist_scale = 2 * self.attention_head_size**1.5
                self.dist_scale = self.attention_head_size

    def forward(self, x, output_attentions=False):
        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a
        num_attention_heads, attention_head_size = self.num_attention_heads, self.attention_head_size
        d_intrinsic = attention_head_size

        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        # if not self.qk_share:            
        #     # Split the projected query, key, and value into query, key, and value
        #     # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        #     query, key, value = torch.chunk(self.qkv_projection(x), 3, dim=-1)
        # else:            
        #     query, value = torch.chunk(self.qv_projection(x), 2, dim=-1)

        if not self.is_op or self.num_attention_heads > 1:
            query = self.WQ(x)
        else:
            query = x
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()    
        query = query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)

        if not self.qk_share:            
            key = self.WK(x)
            key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)            
            g_dist = torch.cdist(query, key, p=2)  # Euclidean dist in R^d             
        else:                                               
            g_dist = torch.cdist(query, query, p=2)
        if self.is_rescale_dist:
            g_dist = g_dist / self.dist_scale

        value = self.WV(x)
        value = value.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        # print(f'query shape: {query.shape}')
        # print(f'key shape: {key.shape}')
        # print(f'value shape: {value.shape}')               
        
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


class SPFNSMultiHeadAttention(nn.Module):
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
        self.qk_share = config['qk_share']
        self.is_op = config["is_op"]

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

        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']
        self.sphere_radius = config['sphere_radius']

    def forward(self, x, output_attentions=False):
        alpha, bandwidth = self.alpha, self.bandwidth
        a = self.a
        sphere_radius = self.sphere_radius
        qk_share = self.qk_share

        num_attention_heads, attention_head_size = self.num_attention_heads, self.attention_head_size
        if num_attention_heads == 1:
            d_intrinsic = attention_head_size - 1
        else:
            d_intrinsic = attention_head_size

        batch_size, sequence_length, _ = x.size()
        if self.is_op:
            x = F.normalize(x,p=2,dim=-1)       
            if not self.qk_share:            
                # Split the projected query, key, and value into query, key, and value
                # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
                #query, key, value = torch.chunk(self.qkv_projection(x), 3, dim=-1)
                key = self.WK(x)     
            if self.num_attention_heads > 1:
                query = self.WQ(x)
                query = F.normalize(query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2), p=2, dim=-1)
                if not self.qk_share:
                    key = F.normalize(key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2), p=2, dim=-1)
            else:
                query = x   
                query = query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)                          
                if not self.qk_share:                
                    key = key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)                       
        else:
            # Project the query, key, and value
            # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)               
            query = self.WQ(x)     
            #batch_size, sequence_length, _ = query.size()
            # Split the projected query, key, and value into query, key, and value
            # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
            if not self.qk_share:            
                # Split the projected query, key, and value into query, key, and value
                # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
                #query, key, value = torch.chunk(self.qkv_projection(x), 3, dim=-1)
                key = self.WK(x)
                key = F.normalize(key.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2), p=2, dim=-1)
            # else:            
            #     query, value = torch.chunk(self.qv_projection(x), 2, dim=-1)                
            # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)            
            query = F.normalize(query.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2), p=2, dim=-1) 

        value = self.WV(x)
        value = value.view(batch_size, sequence_length, num_attention_heads, attention_head_size).transpose(1, 2)
        # print(f'query shape: {query.shape}')
        # print(f'key shape: {key.shape}')
        # print(f'value shape: {value.shape}')        

        # geodesic distance on sphere
        # eps = 1e-7  # for limiting the divergence from acos
        if not self.qk_share:                        
            #g_dist = torch.acos(torch.clamp(query @ key.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius
            g_dist = torch.acos_(torch.matmul(query, key.transpose(-1, -2))) * sphere_radius
        else:
            # geodesic distance on sphere      
            # old method    
            #g_dist = torch.acos(torch.clamp(query @ query.transpose(-2, -1), -1+eps, 1-eps)) * sphere_radius
            # new method
            dot = torch.matmul(query, query.transpose(-1, -2))
            dot = dot.masked_fill_(torch.diag_embed(torch.ones(sequence_length, device=query.device))==1, 1)
            g_dist = torch.acos_(dot) * sphere_radius           

        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))
        attn_score_shape = attn_score.shape
        #print(f'attn_score_shape: {attn_score_shape}')
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