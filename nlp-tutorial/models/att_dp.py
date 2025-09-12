import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(ScaledDotProductAttention, self).__init__()
        #self.d_k = d_k
        self.head_dim = config['head_dim']
        self.device = config['device']
        self.is_return_dist = is_return_dist

        # train mask
        self.train_mask_type = config['train_mask_type']    
        if self.train_mask_type is not None:
            self.train_mask = config['train_mask']         

        # only for evaluation   
        if 'dist_threshold' in config.keys():
            self.dist_threshold = config['dist_threshold']     
        else:
            self.dist_threshold = None
        if 'is_add_eval_mask' in config.keys():
            self.is_add_eval_mask = config['is_add_eval_mask'] 
        else:
            self.is_add_eval_mask = False
        if self.is_add_eval_mask:
            self.eval_mask = config['eval_mask']  

    def forward(self, q, k, v, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        
        batch_size, n_heads, q_len, head_dim = q.size()

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)

        # distance based
        if self.dist_threshold is not None and not self.training:    
            g_dist = torch.cdist(q, k, p=2)        
            #attn_mask = attn_mask | (g_dist > self.dist_threshold)        
            attn_mask = attn_mask | (g_dist >= self.dist_threshold)

        # probability based
        if self.is_add_eval_mask and not self.training:
            attn_mask = attn_mask | self.eval_mask

        # training mask
        if self.train_mask_type is not None:      
            attn_mask = attn_mask | self.train_mask

        attn_score.masked_fill_(attn_mask, -1e9)
        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        attn_weights = nn.Softmax(dim=-1)(attn_score)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        
        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)
        
        if not self.is_return_dist:
            return output, attn_weights
        else:
            g_dist = torch.cdist(q, k, p=2)
            return output, attn_weights, g_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(MultiHeadAttention, self).__init__()                 
        self.n_heads = n_heads = config['n_heads']
        self.d_model = d_model = config['d_model']
        #self.d_k = self.d_v = config['d_model']//self.n_heads
        self.head_dim = config['head_dim']

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']

        self.is_return_dist = is_return_dist

        self.WQ = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        if not self.qk_share:
            self.WK = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.scaled_dot_product_attn = ScaledDotProductAttention(config, is_return_dist)
        self.linear = nn.Linear(n_heads * self.head_dim, d_model)
        
    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        
        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        if not self.qk_share: 
            k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)  
            if not self.is_return_dist:       
                attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            if not self.is_return_dist:
                attn, attn_weights = self.scaled_dot_product_attn(q_heads, q_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.scaled_dot_product_attn(q_heads, q_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)

        if not self.is_return_dist:
            return output, attn_weights
        else:
            return output, attn_weights, g_dist


class OPMultiHeadAttention(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(OPMultiHeadAttention, self).__init__()         
        self.n_heads = n_heads = config['n_heads']
        self.d_model = d_model = config['d_model']
        #self.d_k = self.d_v = config['d_model']//self.n_heads
        self.head_dim = config['head_dim']

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']
        
        self.is_return_dist = is_return_dist

        if not self.qk_share:
            self.WK = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))
        if self.n_heads > 1:
            self.WQ = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))

        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.scaled_dot_product_attn = ScaledDotProductAttention(config, is_return_dist)
        self.linear = nn.Linear(n_heads * self.head_dim, d_model)
        
    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        
        if self.n_heads > 1:
            q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q_heads = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        if not self.qk_share: 
            k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)    
            if not self.is_return_dist:     
                attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            if not self.is_return_dist:
                attn, attn_weights = self.scaled_dot_product_attn(q_heads, q_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.scaled_dot_product_attn(q_heads, q_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)

        if not self.is_return_dist:
            return output, attn_weights
        else:
            return output, attn_weights, g_dist