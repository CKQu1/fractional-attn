import torch
import math
import numpy as np
import operator as operator
from functools import reduce

from models.model_utils import *
from models.utils import *
from dijkstra import DijkstraPQ

from torch import nn
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.nn.utils.parametrizations import orthogonal

from models.sinkhorn import SinkhornDistance

#torch.autograd.detect_anomaly(True)

# -------------------- General classes --------------------

class ModelSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# -------------------- DP-MHA --------------------

class DPAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        self.self = DPSelfAttention(config, layer_id)          
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs

# -------------------- FNS-MHA --------------------

class FNSAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        error_message = f"alpha = {kwargs.get('alpha')} with type {type(kwargs.get('alpha'))} is ill-defined!"
        alpha = kwargs.get('alpha')     
        bandwidth = kwargs.get('bandwidth')             
        assert 0 < alpha < 1, error_message 
        self.self = FNSSelfAttention(config, layer_id, alpha, bandwidth)
     
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs        


class SPFNSAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        error_message = f"alpha = {kwargs.get('alpha')} with type {type(kwargs.get('alpha'))} is ill-defined in SPFNSAttention!"
        alpha = kwargs.get('alpha')
        bandwidth = kwargs.get('bandwidth')
        a = kwargs.get('a')
        #print(f'V3FNSAttention alpha = {alpha}')
        assert 0 < alpha, error_message                           
        self.self = SPFNSSelfAttention(config, layer_id, alpha, bandwidth, a)
     
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs           

class SPOPFNSAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        error_message = f"alpha = {kwargs.get('alpha')} with type {type(kwargs.get('alpha'))} is ill-defined in SPOPFNSAttention!"
        alpha = kwargs.get('alpha')
        bandwidth = kwargs.get('bandwidth')
        a = kwargs.get('a')
        #print(f'V3FNSAttention alpha = {alpha}')
        assert 0 < alpha, error_message                           
        self.self = SPOPFNSSelfAttention(config, layer_id, alpha, bandwidth, a)
     
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs 


# ---------- Return to Euclidean ------

class RDFNSAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        error_message = f"alpha = {kwargs.get('alpha')} with type {type(kwargs.get('alpha'))} is ill-defined in RDFNSAttention!"
        alpha = kwargs.get('alpha')
        bandwidth = kwargs.get('bandwidth')
        a = kwargs.get('a')
        #print(f'V3FNSAttention alpha = {alpha}')
        assert 0 < alpha, error_message                           
        self.self = RDFNSSelfAttention(config, layer_id, alpha, bandwidth, a)
     
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs  

class RDOPFNSAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        error_message = f"alpha = {kwargs.get('alpha')} with type {type(kwargs.get('alpha'))} is ill-defined in RDOPFNSAttention!"
        alpha = kwargs.get('alpha')
        bandwidth = kwargs.get('bandwidth')
        a = kwargs.get('a')
        #print(f'V3FNSAttention alpha = {alpha}')
        assert 0 < alpha, error_message                           
        self.self = RDOPFNSSelfAttention(config, layer_id, alpha, bandwidth, a)
     
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs 

# ---------- Sinkformer ------

class SINKAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        bandwidth = kwargs.get('bandwidth')
        n_it = kwargs.get('n_it')                  
        self.self = SINKSelfAttention(config, layer_id, bandwidth, n_it)
        self.output = ModelSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs  

# -------------------- Original Self-Attention --------------------

class DPSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.qk_share = config.qk_share
        self.bias = config.qkv_bias

        self.query = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        if not self.qk_share:
            self.key = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        #hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states

        query_vectors = self.query(hidden_states)
        if not self.qk_share:
            key_vectors = self.key(hidden_states)        
        value_vectors = self.value(hidden_states)

        #seq_len, batch_size, embed_dim = hidden_states.size()
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)
        if not self.qk_share:
            query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)     # (B,H,N,D)
            value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)            
            att = (query_vectors @ key_vectors.transpose(-2, -1)) * (1.0 / math.sqrt(key_vectors.size(-1)))  # (B,H,N,N)
        else:
            query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)
            value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2) # (B,H,N,D)            
            att = (query_vectors @ query_vectors.transpose(-2, -1)) * (1.0 / math.sqrt(query_vectors.size(-1)))  # (B,H,N,N)
        
        # if attention_mask.dim() == 3:
        #     attention_mask_expanded = attention_mask.view(batch_size, -1, seq_len, seq_len)
        # else:  # attention_mask.dim() == 2
        #     attention_mask_expanded = attention_mask.view(1, 1, seq_len, seq_len).expand(batch_size, num_heads, -1, -1)

        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).long()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)     

            att = att.masked_fill(attention_mask_expanded==0, -1e9)

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # print(f'hidden_states[0]: {hidden_states[0]}')
        # print(f'bool_mask[0]: {bool_mask[0]}')
        # print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')        
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')      
        # print(f'attention_mask[0]: {attention_mask[0]}') 
        # print(f'attention_mask max: {attention_mask.max()}, min: {attention_mask.min()}') 
        # print(f'attention_mask zeros: {(attention_mask==0).sum()}')        
        # print(attention_mask)
        # print('\n')
        # torch.set_printoptions(profile='full')
        # print(torch.diagonal(attention_mask_expanded[0,0]))
        # print(f'att shape: {att.shape}')        
        # print('-'*10)
        #quit()
        # ---------- \end{delete} ---------- 
                            
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)  # attn dropout
        attn_output = att @ value_vectors # (B, H, N, N) x (B, H, N, D) -> (B, H, N, D)        
  
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)  # output dropout
        #assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        assert attn_output.size() == (batch_size, num_heads, seq_len, head_dim), "Unexpected size"
        #attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        #outputs = (attn_output.transpose(0, 1),) # Seq,B,D
        outputs = (attn_output,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs


# -------------------- FNS Self-Attention --------------------

class FNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, alpha, bandwidth):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.qk_share = config.qk_share
        self.bias = config.qkv_bias

        self.query = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        if not self.qk_share:
            self.key = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        self.dropout = config.attention_probs_dropout_prob
        self.alpha = alpha
        self.bandwidth = bandwidth

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        #hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        #seq_len, batch_size, embed_dim = hidden_states.size()
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim
        alpha = self.alpha        
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # -------------------- Strategy 2 --------------------
        """
        Normal embedding
        """

        """
        # normalize query
        #query_vectors /= math.sqrt(head_dim)

        # weight-share between query and key
        query_vectors = query_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) #B,N,H,D
        
        # normalize query row-wise (embedded in D-sphere)
        #query_vectors =  query_vectors.reshape(-1, num_heads, head_dim)  #BN,H,D
        query_vectors =  query_vectors.reshape(num_heads, -1, head_dim)  # (H,BN,D)
        query_vectors = normalize(query_vectors, p=2, dim=2)
        value_vectors =  value_vectors.reshape(-1, num_heads, head_dim)  #BN,H,D  
        """

        # ----------
        # (B, N, H, D)
        # (batch_size, seq_len, num_heads, head_dim)
        # ----------

        query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,H,N,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,H,N,D)   
        if not self.qk_share:      
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B,H,N,D)                         
            euclidean_dist = torch.cdist(query_vectors, key_vectors) # (B,H,N,N) pairwise Euclidean distance   
        else:                         
            euclidean_dist = torch.cdist(query_vectors, query_vectors) # (B,H,N,N)     

        # attention_mask obtained from class Model in models/model.py    

        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).long()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)   

            #euclidean_dist = euclidean_dist.masked_fill(attention_mask_expanded==0, float('inf'))  # CHECK!!!  
            euclidean_dist = euclidean_dist.masked_fill(attention_mask_expanded==0, 1e9)

        # bandwidth set up 1   
        #bandwidth = torch.max(euclidean_dist) + 1 # (H)
        # bandwidth set up 2 (max distance on the surface of a D-sphere)      
        
        dijk = DijkstraPQ()
        g_dist = dijk(euclidean_dist)
        attn_score = (1 + g_dist/self.bandwidth**0.5)**(-head_dim - alpha)                  
        attn_score_shape = attn_score.shape
        #attn_score = attn_score.view(-1, attn_score_shape[2], attn_score_shape[3])
        attn_weights = nn.Softmax(dim=-1)(attn_score)      
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)   

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(f'attn_score shape: {attn_score.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # print(attn_weights.sum(-1))
        # print('-'*10)
        # ---------- \end{delete} ----------  

        # -----------------------------------------------------
        #attn_output = torch.bmm(attn_weights, value_vectors)
        attn_output = attn_weights @ value_vectors        

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs

# -------------------- SP: FNS Self-Attention --------------------
"""
- embedded in sphere
- specify normalization index a
- use column normalization
"""

class SPFNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, alpha, bandwidth, a):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.alpha = alpha
        self.bandwidth = bandwidth
        self.a = a

        self.qk_share = config.qk_share
        self.sphere_radius = config.sphere_radius
        self.mask_val = config.mask_val  # mask for g_dist
        self.bias = config.qkv_bias

        self.query = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        if not self.qk_share:          
            self.key = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        if self.alpha < 2:
            self.d_intrinsic = config.d_intrinsic  # should still be self.head_dim
            error_message = f'Incorrect: hidden_size = {config.hidden_size}, d_intrinsic = {self.d_intrinsic}, num_heads = {self.num_heads} in SPFNSSelfAttention'
            assert config.hidden_size == (self.d_intrinsic + 1) * self.num_heads, error_message

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        alpha = self.alpha
        bandwidth = self.bandwidth
        a = self.a

        if alpha < 2:
            d_intrinsic = self.d_intrinsic

        hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        query_vectors = self.query(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        #seq_len, batch_size, embed_dim = hidden_states.size()        
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim                
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        """
        Embed the queries into a D-sphere via normalization, set the bandwidth as the diamter of an D-sphere
        """   
        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        query_vectors = F.normalize(query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)  # (B,N,H,D)
        #query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        eps = 1e-7  # for limiting the divergence from acos
        if not self.qk_share:
            key_vectors = self.key(hidden_states)    
            key_vectors = F.normalize(key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)      # (B,H,N,D)  
            # directly get geodesic distance
            g_dist = torch.acos(torch.clamp(query_vectors @ key_vectors.transpose(-2, -1), -1+eps, 1-eps)) * self.sphere_radius
        else:
            g_dist = torch.acos(torch.clamp(query_vectors @ query_vectors.transpose(-2, -1), -1+eps, 1-eps)) * self.sphere_radius                        

        # obtained from class Model in models/model.py
        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).int()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).int()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)      

            g_dist = g_dist.masked_fill(attention_mask_expanded==0, self.mask_val)  # 1e9

        if alpha < 2:
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))

        attn_score_shape = attn_score.shape        
        if a > 0:
            # if self.qk_share:
            #     D_inv = torch.diag_embed(attn_score.sum(-1)**(-a))  # inverse of degree matrix of attn_score
            #     K_tilde = D_inv @ attn_score @ D_inv
            # else:
            K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            # N_R = attn_score.sum(-1)  # row sum
            # N_C = attn_score.sum(-2)  # col su                
            # K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)            

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive
        
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)   

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # if not self.qk_share:
        #     print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'g_dist shape: {g_dist.shape}')
        # #print(g_dist)
        # print(f'g_dist nans: {torch.isnan(g_dist.view(-1)).sum()}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(f'attn_score shape: {attn_score.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # #print(attn_weights.sum(-1))
        # print('-'*10)
        # ---------- \end{delete} ----------  

        attn_output = attn_weights @ value_vectors        

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs      


# -------------------- RD: FNS Self-Attention (R^d manifold) --------------------

class RDFNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, alpha, bandwidth, a):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.alpha = alpha
        self.bandwidth = bandwidth
        self.a = a

        self.mask_val = config.mask_val  # mask for attn_score

        self.qk_share = config.qk_share
        self.bias = config.qkv_bias

        self.query = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        if not self.qk_share:          
            self.key = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        if self.alpha < 2:
            self.d_intrinsic = config.d_intrinsic  # should still be self.head_dim
            assert config.hidden_size == self.d_intrinsic * self.num_heads, 'Incorrect d_intrinsic in RDFNSSelfAttention'

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        alpha = self.alpha
        bandwidth = self.bandwidth
        a = self.a

        mask_val = self.mask_val  # mask for attn_score

        if alpha < 2:
            d_intrinsic = self.d_intrinsic

        query_vectors = self.query(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        # begin{Lipschitz-MHA}
        # if not self.qk_share:
        #     XA = query_vectors @ self.key.weight
        # else:
        #     XA = query_vectors @ hself.query.weight
        # end{Lipschitz-MHA}

        #seq_len, batch_size, embed_dim = hidden_states.size()
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim                
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        """
        Embed the queries into a D-sphere via normalization, set the bandwidth as the diamter of an D-sphere
        """   
        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)
        #query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

        # begin{Lipschitz-MHA}
        #AXW_V = self.value(XA).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        # end{Lipschitz-MHA}

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        if not self.qk_share:
            key_vectors = self.key(hidden_states)  
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B,H,N,D)              
            # directly get Euclidean dist
            # g_dist = torch.cdist(query_vectors, key_vectors, p=2)  # (H,B,N,N)
            g_dist = torch.cdist(query_vectors, key_vectors, p=2)  # (H,B,N,N)
        else:
            # directly get Euclidean dist
            g_dist = torch.cdist(query_vectors, query_vectors, p=2)  # (H,B,N,N)                       
        #print(f'Min and max distance: {g_dist.min()}, {g_dist.max()}')  
        #q = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1])
        #print(f'Distance percentiles: {torch.quantile(g_dist.flatten(), q)}')    

        # g_dist = g_dist.masked_fill(attention_mask_expanded==0, 1e5)

        if alpha < 2:            
            #attn_score = (1 + (d_intrinsic**(1/d_intrinsic) - 1) / math.sqrt(d_intrinsic) * g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
            attn_score = (1 + 1 / math.sqrt(d_intrinsic) * g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
            # attn_score = (1 + g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
        else:             
            attn_score = torch.exp(-(g_dist / head_dim**0.5 / bandwidth**0.5)**(alpha/(alpha-1)))
            # attn_score = torch.exp(-(g_dist / bandwidth**0.5)**(alpha/(alpha-1)))

        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).long()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1) 

            attn_score = attn_score.masked_fill(attention_mask_expanded==0, mask_val)

        attn_score_shape = attn_score.shape
        #bound = 1e9 * seq_len
        bound = 1e5
        if a > 0:
            # if self.qk_share:                
            #     # D_inv = torch.diag_embed(attn_score.sum(-1)**(-a))  # inverse of degree matrix of attn_score
            #     # K_tilde = D_inv @ attn_score @ D_inv
            #     #N_R = torch.clamp(attn_score.sum(-1), min=1/bound, max=bound)  # row sum
            #     N_R = attn_score.sum(-1)  # row sum
            #     K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_R**(-a)).unsqueeze(-2)
            # else:
            #     # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            #     # N_R = torch.clamp(attn_score.sum(-1), min=1/bound, max=bound)  # row sum
            #     # N_C = torch.clamp(attn_score.sum(-2), min=1/bound, max=bound)  # col sum
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive     
        
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)   

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # if not self.qk_share:
        #     print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'g_dist shape: {g_dist.shape}')
        # #print(g_dist)
        # print(f'g_dist nans: {torch.isnan(g_dist.view(-1)).sum()}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(f'attn_score shape: {attn_score.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # #print(attn_weights.sum(-1))
        # print('-'*10)
        # ---------- \end{delete} ----------  

        #quit()  # delete

        # -----------------------------------------------------
        attn_output = attn_weights @ value_vectors        

        # begin{Lipschitz-MHA}
        #attn_output = attn_weights @ AXW_V
        # end{Lipschitz-MHA}

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs      


# -------------------- OPFNS Self-Attention (old version)  --------------------
# Orthogonal Projection: same as V3FNSSelfAttention but with orthogonal projections for QK, not for V
"""
class SPOPFNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, alpha, bandwidth, a):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.alpha = alpha
        self.bandwidth = bandwidth
        self.a = a

        self.qk_share = config.qk_share
        self.sphere_radius = config.sphere_radius
        self.mask_val = config.mask_val  # mask for g_dist
        self.bias = config.qkv_bias

        self.query = orthogonal(nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias))
        if not self.qk_share:          
            self.key = orthogonal(nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias))
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        if self.alpha < 2:
            self.d_intrinsic = config.d_intrinsic  
            assert config.hidden_size == (self.d_intrinsic + 1) * self.num_heads, 'Incorrect d_intrinsic in SPOPFNSSelfAttention'

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        alpha = self.alpha
        bandwidth = self.bandwidth
        a = self.a

        if alpha < 2:
            d_intrinsic = self.d_intrinsic

        hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        query_vectors = self.query(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        #seq_len, batch_size, embed_dim = hidden_states.size()        
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim                
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        
        # Embed the queries into a D-sphere via normalization, set the bandwidth as the diamter of an D-sphere
           
        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        query_vectors = F.normalize(query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)  # (B,N,H,D)
        #query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        eps = 1e-7  # for limiting the divergence from acos
        if not self.qk_share:
            key_vectors = self.key(hidden_states)    
            key_vectors = F.normalize(key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)      # (B,H,N,D)  
            #g_dist = torch.cdist(query_vectors, key_vectors, p=2)  # (H,B,N,N)
            # directly get geodesic distance
            g_dist = torch.acos(torch.clamp(query_vectors @ key_vectors.transpose(-2, -1), -1+eps, 1-eps)) * self.sphere_radius
        else:
            #g_dist = torch.cdist(query_vectors, query_vectors, p=2)  # (H,B,N,N)      
            # directly get geodesic distance
            g_dist = torch.acos(torch.clamp(query_vectors @ query_vectors.transpose(-2, -1), -1+eps, 1-eps)) * self.sphere_radius                        
        #print(f'Min and max distance: {g_dist.min()}, {g_dist.max()}')  
        #q = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1])
        #print(f'Distance percentiles: {torch.quantile(g_dist.flatten(), q)}')


        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).long()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)      

            g_dist = g_dist.masked_fill(attention_mask_expanded==0, self.mask_val)  # 1e9

        if alpha < 2:
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))

        attn_score_shape = attn_score.shape
        #attn_score = attn_score.view(-1, attn_score_shape[2], attn_score_shape[3])
        if a > 0:
            # if self.qk_share:
            #     D_inv = torch.diag_embed(attn_score.sum(-1)**(-a))  # inverse of degree matrix of attn_score
            #     K_tilde = D_inv @ attn_score @ D_inv
            # else:
            # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)   

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive         
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive
        
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)   

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # if not self.qk_share:
        #     print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'g_dist shape: {g_dist.shape}')
        # #print(g_dist)
        # print(f'g_dist nans: {torch.isnan(g_dist.view(-1)).sum()}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(f'attn_score shape: {attn_score.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # #print(attn_weights.sum(-1))
        # print('-'*10)
        # ---------- \end{delete} ----------  

        #quit()  # delete

        # -----------------------------------------------------
        attn_output = attn_weights @ value_vectors        

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs      
        """

# -------------------- OPFNS Self-Attention (new version)  --------------------
# Orthogonal Projection: same as V3FNSSelfAttention but with orthogonal projections for QK, not for V

class SPOPFNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, alpha, bandwidth, a):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.alpha = alpha
        self.bandwidth = bandwidth
        self.a = a

        self.qk_share = config.qk_share
        self.sphere_radius = config.sphere_radius
        self.mask_val = config.mask_val  # mask for g_dist
        self.bias = config.qkv_bias

        #self.query = orthogonal(nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias))
        if not self.qk_share:          
            self.key = orthogonal(nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias))
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        if self.alpha < 2:
            self.d_intrinsic = config.d_intrinsic  
            assert config.hidden_size == (self.d_intrinsic + 1) * self.num_heads, 'Incorrect d_intrinsic in SPOPFNSSelfAttention'

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        alpha = self.alpha
        bandwidth = self.bandwidth
        a = self.a

        if alpha < 2:
            d_intrinsic = self.d_intrinsic

        hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        #query_vectors = self.query(hidden_states)
        query_vectors = hidden_states
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        #seq_len, batch_size, embed_dim = hidden_states.size()        
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim                
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        """
        Embed the queries into a D-sphere via normalization, set the bandwidth as the diamter of an D-sphere
        """   
        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        #query_vectors = F.normalize(query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)  # (B,N,H,D)
        query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        eps = 1e-7  # for limiting the divergence from acos
        if not self.qk_share:
            key_vectors = self.key(hidden_states)    
            #key_vectors = F.normalize(key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)      # (B,H,N,D)  
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B,H,N,D)
            # directly get geodesic distance
            g_dist = torch.acos(torch.clamp(query_vectors @ key_vectors.transpose(-2, -1), -1+eps, 1-eps)) * self.sphere_radius
        else:   
            # directly get geodesic distance            
            g_dist = torch.acos(torch.clamp(query_vectors @ query_vectors.transpose(-2, -1), -1+eps, 1-eps)) * self.sphere_radius                        

        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).long()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)      

            g_dist = g_dist.masked_fill(attention_mask_expanded==0, self.mask_val)  # 1e9

        if alpha < 2:
            attn_score = (1 + g_dist/bandwidth**0.5)**(-d_intrinsic-alpha)
        else:
            attn_score = torch.exp(-(g_dist/bandwidth**0.5)**(alpha/(alpha-1)))

        attn_score_shape = attn_score.shape
        #attn_score = attn_score.view(-1, attn_score_shape[2], attn_score_shape[3])
        if a > 0:
            # if self.qk_share:
            #     D_inv = torch.diag_embed(attn_score.sum(-1)**(-a))  # inverse of degree matrix of attn_score
            #     K_tilde = D_inv @ attn_score @ D_inv
            # else:
            # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)   

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive         
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive
        
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)   

        # -----------------------------------------------------
        attn_output = attn_weights @ value_vectors        

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs     



# -------------------- RDOPFNS Self-Attention (stick with R^d)  --------------------
# Orthogonal Projection: same as RDFNSSelfAttention but with orthogonal projections for QK, not for V

class RDOPFNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, alpha, bandwidth, a):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.alpha = alpha
        self.bandwidth = bandwidth
        self.a = a

        self.mask_val = config.mask_val  # mask for attn_score

        self.qk_share = config.qk_share
        self.bias = config.qkv_bias

        self.query = orthogonal(nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias))
        if not self.qk_share:           
            self.key = orthogonal(nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias))        
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        # embed query/key into lower dim
        if self.alpha < 2:
            self.d_intrinsic = config.d_intrinsic            
            assert config.hidden_size == self.d_intrinsic * self.num_heads, 'Incorrect d_intrinsic in RDOPFNSSelfAttention'                   

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        alpha = self.alpha
        bandwidth = self.bandwidth
        a = self.a

        mask_val = self.mask_val

        if alpha < 2:
            d_intrinsic = self.d_intrinsic

        query_vectors = self.query(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        #seq_len, batch_size, embed_dim = hidden_states.size()
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim                
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        """
        Embed the queries into a D-sphere via normalization, set the bandwidth as the diamter of an D-sphere
        """   
        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)
        #query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)                 

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        if not self.qk_share:
            key_vectors = self.key(hidden_states) 
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B,H,N,D)              
            # directly get geodesic distance
            g_dist = torch.cdist(query_vectors, key_vectors, p=2)  # (H,B,N,N)
        else:
            # directly get geodesic distance
            g_dist = torch.cdist(query_vectors, query_vectors, p=2)  # (H,B,N,N)                                       
        #print(f'Min and max distance: {g_dist.min()}, {g_dist.max()}')  
        #q = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1])
        #print(f'Distance percentiles: {torch.quantile(g_dist.flatten(), q)}')  

        if alpha < 2:            
            #attn_score = (1 + (d_intrinsic**(1/d_intrinsic) - 1) / math.sqrt(d_intrinsic) * g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
            #attn_score = (1 + 1 / math.sqrt(d_intrinsic) * g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
            attn_score = (1 + g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
        else:             
            #attn_score = torch.exp(-(g_dist / head_dim**0.5 / bandwidth**0.5)**(alpha/(alpha-1)))
            attn_score = torch.exp(-(g_dist / bandwidth**0.5)**(alpha/(alpha-1)))            

        if attention_mask is not None:
            # type 1: key_pad_mask
            bool_mask = (attention_mask>=0).long()
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)   

            attn_score = attn_score.masked_fill(attention_mask_expanded==0, mask_val)

        attn_score_shape = attn_score.shape
        #attn_score = attn_score.view(-1, attn_score_shape[2], attn_score_shape[3])
        # min_bound = 1e-9
        if a > 0:
            # if self.qk_share:
            #     # D_inv = torch.diag_embed(attn_score.sum(-1)**(-a))  # inverse of degree matrix of attn_score
            #     # K_tilde = D_inv @ attn_score @ D_inv
            #     N_R = torch.clamp(attn_score.sum(-1), min=min_bound, max=None)  # row sum
            #     K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_R**(-a)).unsqueeze(-2)
            # else:
            # K_tilde = torch.diag_embed(attn_score.sum(-1)**(-a)) @ attn_score @ torch.diag_embed(attn_score.sum(-2)**(-a))
            # N_R = torch.clamp(attn_score.sum(-1), min=min_bound, max=None)  # row sum
            # N_C = torch.clamp(attn_score.sum(-2), min=min_bound, max=None)  # col sum
            # K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)            

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive 
        
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)   

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # if not self.qk_share:
        #     print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'g_dist shape: {g_dist.shape}')
        # #print(g_dist)
        # print(f'g_dist nans: {torch.isnan(g_dist.view(-1)).sum()}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(f'attn_score shape: {attn_score.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # #print(attn_weights.sum(-1))
        # print('-'*10)
        # ---------- \end{delete} ----------  

        #quit()  # delete

        # -----------------------------------------------------
        attn_output = attn_weights @ value_vectors        

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs      


# -------------------- Sinkformer Self-Attention  --------------------

class SINKSelfAttention(nn.Module):
    def __init__(self, config, layer_id, bandwidth, n_it):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.n_it = n_it
        self.bandwidth = bandwidth
        self.qk_share = config.qk_share
        self.mask_val = config.mask_val  # mask for attn_score
        self.bias = config.qkv_bias

        self.query = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        if not self.qk_share:
            self.key = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=self.bias)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        query_vectors = self.query(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim                
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        n_it = self.n_it; bandwidth = self.bandwidth
        mask_val = self.mask_val                  

        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        
        query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,N,H,D)        
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,H,N,D)                 

        if not self.qk_share:
            key_vectors = self.key(hidden_states) 
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B,H,N,D)  
            attn_score = query_vectors @ key_vectors.transpose(-2, -1) / np.sqrt(head_dim)
        else:
            attn_score = query_vectors @ query_vectors.transpose(-2, -1) / np.sqrt(head_dim)   

        if attention_mask is not None:
            # type 1: key_pad_mask
            #bool_mask = (attention_mask>=0).long()
            bool_mask = (attention_mask>=0)
            attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_heads,1,-1])

            # type 2: symmetrical mask
            # bool_mask = (attention_mask>=0).long()
            # attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1) 
        
            attn_score.masked_fill_(attention_mask_expanded, mask_val)

        attn_score_shape = attn_score.shape

        attn_weights_soft = nn.Softmax(dim=-1)(attn_score)
        attn_score = attn_score.view(-1, attn_score_shape[2], attn_score_shape[3])  # (B,H,N,D)
        sink = SinkhornDistance(self.bandwidth, max_iter=n_it)
        attn_weights = sink(attn_score)[0]

        attn_weights = attn_weights * attn_weights.shape[-1]
        attn_weights = attn_weights.view(attn_score_shape)  # (B,H,N,D)        

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # if not self.qk_share:
        #     print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'attn_score shape: {attn_score.shape}')
        # print(attn_score)
        # print(f'attn_score nans: {torch.isnan(attn_score.view(-1)).sum()}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(f'attn_score shape: {attn_score.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # print(attn_weights.sum(-1))
        # print('-'*10)
        # ---------- \end{delete} ----------  
        #quit()  # delete

        # -----------------------------------------------------
        attn_output = attn_weights @ value_vectors  # (B, H, N, D_v)   
        #attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)  # not used in Sinkformer for nlp-classification
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs            