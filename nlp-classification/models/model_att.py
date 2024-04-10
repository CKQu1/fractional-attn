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
        error_message = f"beta = {kwargs.get('beta')} with type {type(kwargs.get('beta'))} is ill-defined!"
        beta = kwargs.get('beta')     
        bandwidth = kwargs.get('bandwidth')             
        assert 0 < beta < 1, error_message 
        self.self = FNSSelfAttention(config, layer_id, beta, bandwidth)
     
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


class V2FNSAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        error_message = f"beta = {kwargs.get('beta')} with type {type(kwargs.get('beta'))} is ill-defined!"
        beta = kwargs.get('beta')
        bandwidth = kwargs.get('bandwidth')
        #print(f'V2FNSAttention beta = {beta}')
        assert 0 < beta < 1, error_message                           
        self.self = V2FNSSelfAttention(config, layer_id, beta, bandwidth)
     
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

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        if not self.qk_share:
            self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

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
            att = (query_vectors @ query_vectors.transpose(-2, -1)) * (1.0 / math.sqrt(key_vectors.size(-1)))  # (B,H,N,N)
        
        # if attention_mask.dim() == 3:
        #     attention_mask_expanded = attention_mask.view(batch_size, -1, seq_len, seq_len)
        # else:  # attention_mask.dim() == 2
        #     attention_mask_expanded = attention_mask.view(1, 1, seq_len, seq_len).expand(batch_size, num_heads, -1, -1)

        # obtained from class Model in models/model.py
        bool_mask = (attention_mask>=0).long()
        #attention_mask_expanded = torch.bmm(bool_mask.unsqueeze(-1), bool_mask.unsqueeze(1))
        attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)     

        # ---------- \begin{delete} ----------
        # print('-'*10)
        # print(f'batch_size: {batch_size}, seq_len: {seq_len}, embed_dim: {embed_dim}')
        # print(f'key_vectors shape: {key_vectors.shape}')
        # print(f'query_vectors shape: {query_vectors.shape}')
        # print(f'value_vectors shape: {value_vectors.shape}')
        # print(f'attention_mask shape: {attention_mask.shape}')        
        # print(f'attention_mask_expanded shape: {attention_mask_expanded.shape}')        
        # print(attention_mask)
        # print('\n')
        # torch.set_printoptions(profile='full')
        # print(torch.diagonal(attention_mask_expanded[0,0]))
        # print(f'att shape: {att.shape}')        
        # print('-'*10)
        # quit()
        # ---------- \end{delete} ---------- 
                    
        #att = att.masked_fill(attention_mask_expanded==0, float('-inf'))  # CHECK!!!
        att = att.masked_fill(attention_mask_expanded==0, -1e9)
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
    def __init__(self, config, layer_id, beta, bandwidth):
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

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        if not self.qk_share:
            self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob
        self.beta = beta
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
        beta = self.beta        
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
        bool_mask = (attention_mask>=0).long()
        #attention_mask_expanded = torch.bmm(bool_mask.unsqueeze(-1), bool_mask.unsqueeze(1))
        attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)      

        #euclidean_dist = euclidean_dist.masked_fill(attention_mask_expanded==0, float('inf'))  # CHECK!!!  
        euclidean_dist = euclidean_dist.masked_fill(attention_mask_expanded==0, 1e9)
        # bandwidth set up 1   
        #bandwidth = torch.max(euclidean_dist) + 1 # (H)
        # bandwidth set up 2 (max distance on the surface of a D-sphere)      
        
        dijk = DijkstraPQ()
        g_dist = dijk(euclidean_dist)
        attn_score = (1 + g_dist/self.bandwidth**0.5)**(-head_dim - beta)                  
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


# -------------------- FNS Self-Attention with sphererical embedding of query/key --------------------

class V2FNSSelfAttention(nn.Module):
    def __init__(self, config, layer_id, beta, bandwidth):
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

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        if not self.qk_share:
            self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob
        self.beta = beta
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
        if not self.qk_share:
            key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        #seq_len, batch_size, embed_dim = hidden_states.size()
        batch_size, seq_len, embed_dim = hidden_states.size()
        num_heads, head_dim = self.num_heads, self.head_dim
        beta = self.beta        
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # -------------------- Strategy 1 --------------------
        """
        Embed the queries into a D-sphere via normalization, set the bandwidth as the diamter of an D-sphere
        """
        # (B, N, H, D) = (batch_size, seq_len, num_heads, head_dim)        

        # normalize query/key
        #query_vectors = F.normalize(query_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)  # (B,N,H,D)
        query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)                                            # (B,N,H,D)
        #key_vectors = F.normalize(key_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2), p=2, dim=-1)      # (B,H,N,D)
        value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)                            # (B,H,N,D)   

        # correlation matrix for each attn head (seems more correct)        
        # with torch.no_grad():
        #     q = query_vectors.reshape(-1, num_heads, head_dim).tranpose(0,1).detach()  # (H,BN,D)
        #     q_mean = q.mean(1)  # (H,D)
        #     C = q.tranpose(1,2) @ q / q.shape(1) - q_mean.unsqueeze(-1) @ q_mean.unsqueeze(1)  # (H,D,D)             
        #     d_intrinsic = ((torch.diagonal(C,dim1=1,dim2=2)).sum(1))**2 - (torch.diagonal(C@C,dim1=1,dim2=2)).sum(1)  # (H)
        #     d_intrinsic = d_intrinsic.detach()        

        # correlation matrix
        # with torch.no_grad():
        #     q = query_vectors.reshape(-1, head_dim).detach()  # (HBN,D)
        #     #print(f'q shape 0: {q.shape[0]}')
        #     q_mean = q.mean(0)  # (D)
        #     #print(f'q_mean shape: {q_mean.shape}')
        #     C = q.T @ q / (batch_size*seq_len*num_heads) - q_mean.unsqueeze(-1) @ q_mean.unsqueeze(0)  # (D,D)             
        #     d_intrinsic = (torch.diagonal(C).sum())**2/(torch.diagonal(C@C)).sum()  # (1)
        #     #print(d_intrinsic)  # delete
        #     d_intrinsic = d_intrinsic.detach()     
        
        d_intrinsic = 3
        #d_intrinsic = np.sqrt(head_dim)
        query_vectors = query_vectors.transpose(1,2)

        # pairwise Euclidean distance (H,BN,D) @ (H,D,BN)
        euclidean_dist = torch.cdist(query_vectors, query_vectors, p=2)  # (H,B,N,N)      
        #print(f'Min and max distance: {euclidean_dist.min()}, {euclidean_dist.max()}')  
        #q = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1])
        #print(f'Distance percentiles: {torch.quantile(euclidean_dist.flatten(), q)}')

        # obtained from class Model in models/model.py
        bool_mask = (attention_mask>=0).long()
        attention_mask_expanded = (bool_mask.unsqueeze(-1)@bool_mask.unsqueeze(1)).view(batch_size, 1, seq_len, seq_len).expand(-1, num_heads, -1, -1)      

        #euclidean_dist = euclidean_dist.masked_fill(attention_mask_expanded==0, 1e9)
        euclidean_dist = euclidean_dist.masked_fill(attention_mask_expanded==0, 2)  # further is capped by 2 on an d-sphere
        
        # On a sphere, we have g_dist = euclidean_dist
        #attn_score = (1 + euclidean_dist/self.bandwidth**0.5)**(-head_dim - beta)                  
        #attn_score = (1 + euclidean_dist/self.bandwidth**0.5)**(-beta)
        attn_score = (1 + euclidean_dist/self.bandwidth**0.5)**(-d_intrinsic - beta)

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
        attn_output = attn_weights @ value_vectors        

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs        


