import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

class FNSSelfAttention(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(FNSSelfAttention, self).__init__()
        self.head_dim = config['head_dim']
        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']    
        self.is_rescale_dist = config['is_rescale_dist']
        self.qk_share = config['qk_share']

        self.mask_val = config['mask_val']

        self.is_return_dist = is_return_dist
        self.device = config['device']

        # dependence on d for non-local kernel   
        if self.alpha < 2: 
            self.d_intrinsic = config['d_intrinsic']

        if self.is_rescale_dist:
            self.sphere_radius = config['sphere_radius']

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
        #with torch.autograd.set_detect_anomaly(True):
        alpha, bandwidth, a = self.alpha, self.bandwidth, self.a
        batch_size, n_heads, q_len, head_dim = q.size()
        if self.alpha < 2: 
            d_intrinsic = self.d_intrinsic

        # dot-product
        dot = q @ k.transpose(-2, -1)
        # eps = 1e-7
        dot = dot.masked_fill_(dot>=1, 1)
        dot = dot.masked_fill_(dot<=-1, -1)  

        if self.is_rescale_dist:                        
            g_dist = torch.acos_(dot) * self.sphere_radius            
        else:            
            g_dist = torch.acos_(dot)
        g_dist.masked_fill_(attn_mask, self.mask_val)  # special token mask
            
        if alpha < 2:                      
            attn_score = (1 + g_dist / bandwidth**(1/alpha))**(-d_intrinsic-alpha)
        else:             
            attn_score = torch.exp(-(g_dist / bandwidth**(1/alpha))**(alpha/(alpha-1)))    

        if self.qk_share:  # Q = K
            attn_score = attn_score.masked_fill(torch.diag_embed(torch.ones(q_len, device=q.device))==1, 0)

        # distance based
        if self.dist_threshold is not None and not self.training:            
            #attn_mask = attn_mask | (g_dist > self.dist_threshold)        
            attn_mask = attn_mask | (g_dist >= self.dist_threshold)

        # probability based
        if self.is_add_eval_mask and not self.training:
            attn_mask = attn_mask | self.eval_mask

        # training mask
        if self.train_mask_type is not None:      
            attn_mask = attn_mask | self.train_mask

        #attn_score = attn_score.masked_fill(attn_mask, 1e-9)
        attn_score = attn_score.masked_fill(attn_mask, 0)
        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        #attn_weights = nn.Softmax(dim=-1)(attn_score)
        if a > 0:
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col sum                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)            

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive 
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        
        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)
        
        if not self.is_return_dist:
            return output, attn_weights
        else:
            return output, attn_weights, g_dist

class SPFNSAttention(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(SPFNSAttention, self).__init__()

        #self.d_k = self.d_v = config['d_model']//self.n_heads
        self.head_dim = head_dim = config['head_dim']
        self.n_heads = n_heads = config['n_heads']
        self.d_model = d_model = config['d_model']        

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']

        self.is_return_dist = is_return_dist

        self.WQ = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        if not self.qk_share:
            self.WK = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.fns_attn = FNSSelfAttention(config, is_return_dist)
        self.linear = nn.Linear(n_heads * head_dim, d_model)
        
    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        
        q_heads = F.normalize(self.WQ(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2),
                              p=2,dim=-1) 
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        if not self.qk_share:
            k_heads = F.normalize(self.WK(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2),
                                  p=2,dim=-1)     
            if not self.is_return_dist:    
                attn, attn_weights = self.fns_attn(q_heads, k_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.fns_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            if not self.is_return_dist:
                attn, attn_weights = self.fns_attn(q_heads, q_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.fns_attn(q_heads, q_heads, v_heads, attn_mask)

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

class OPSPFNSAttention(nn.Module):
    def __init__(self, config, is_return_dist=False):
        super(OPSPFNSAttention, self).__init__()         
        self.n_heads = n_heads = config['n_heads']
        self.d_model = d_model = config['d_model']
        #self.d_k = self.d_v = config['d_model']//self.n_heads
        self.head_dim = head_dim = config['head_dim']

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']

        self.is_return_dist = is_return_dist
        
        if not self.qk_share:
            self.WK = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))
        if self.n_heads > 1:
            self.WQ = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))

        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        #self.WV = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))
        self.fns_attn = FNSSelfAttention(config, is_return_dist)
        self.linear = nn.Linear(n_heads * head_dim, d_model)
        
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
                attn, attn_weights = self.fns_attn(q_heads, k_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.fns_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            if not self.is_return_dist:
                attn, attn_weights = self.fns_attn(q_heads, q_heads, v_heads, attn_mask)
            else:
                attn, attn_weights, g_dist = self.fns_attn(q_heads, q_heads, v_heads, attn_mask)

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