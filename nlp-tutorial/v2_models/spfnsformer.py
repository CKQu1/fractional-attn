import numpy as np
import torch
import torch.nn as nn

import numpy as np
from torch.nn import functional as F

class FNSSelfAttention(nn.Module):
    def __init__(self, config):
        super(FNSSelfAttention, self).__init__()
        #self.d_k = d_k
        self.head_dim = config['head_dim']
        self.alpha = config['alpha']
        self.bandwidth = config['bandwidth']
        self.a = config['a']  

        self.sphere_radius = config['sphere_radius'] 
        self.mask_val = config['mask_val']
    
    def forward(self, q, k, v, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        
        alpha = self.alpha
        bandwidth = self.bandwidth
        a = self.a        
        _, _, n_heads, head_dim = q.size()
        d_intrinsic = head_dim

        sphere_radius = self.sphere_radius
        mask_val = self.mask_val   

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        g_dist = torch.acos_(q @ k.transpose(-2, -1)) * sphere_radius
        g_dist.masked_fill_(attn_mask, mask_val)

        if alpha < 2:            
            #attn_score = (1 + g_dist / head_dim**0.5 / bandwidth**0.5)**(-d_intrinsic-alpha)
            attn_score = (1 + g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
        else:             
            #attn_score = torch.exp(-(g_dist / head_dim**0.5 / bandwidth**0.5)**(alpha/(alpha-1)))
            attn_score = torch.exp(-(g_dist / bandwidth**0.5)**(alpha/(alpha-1)))    
        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        #attn_weights = nn.Softmax(dim=-1)(attn_score)
        if a > 0:
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col su                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)            

            attn_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:
            attn_weights = F.normalize(attn_score,p=1,dim=3)  # can do this as the attn weights are always positive 

        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        
        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)
        
        return output, attn_weights

class FNSAttention(nn.Module):
    def __init__(self, config):
        super(FNSAttention, self).__init__()

        #self.d_k = self.d_v = config['d_model']//self.n_heads
        self.head_dim = head_dim = config['head_dim']
        self.n_heads = n_heads = config['n_heads']
        self.d_model = d_model = config['d_model']        

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']

        self.WQ = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        if not self.qk_share:
            self.WK = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.fns_attn = FNSSelfAttention(config)
        self.linear = nn.Linear(n_heads * head_dim, d_model)
        
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
            attn, attn_weights = self.fns_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            attn, attn_weights = self.fns_attn(q_heads, q_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)

        return output, attn_weights

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model = config['d_model']
        self.n_heads = n_heads = config['n_heads']
        self.p_drop = p_drop = config['p_drop']
        self.d_ff = d_ff = config['d_ff']        

        self.mha = FNSAttention(config)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        # ----- added -----
        inputs = x = F.normalize(inputs,p=2,dim=-1)

        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        return ffn_outputs, attn_weights

class SPFNSformer(nn.Module):
    
    def __init__(self, config):
        super(SPFNSformer, self).__init__()

        self.vocab_size = config['vocab_size']
        self.seq_len = config['seq_len']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.p_drop = config['p_drop']
        self.d_ff = config['d_ff']
        self.pad_id = config['pad_id']

        self.sinusoid_table = self.get_sinusoid_table(self.seq_len+1, self.d_model) # (seq_len+1, d_model)

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(self.n_layers)])

        # layers to classify
        self.linear = nn.Linear(self.d_model, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attention_weights.append(attn_weights)

        outputs, _ = torch.max(outputs, dim=1)
        # |outputs| : (batch_size, d_model)
        outputs = self.softmax(self.linear(outputs))
        # |outputs| : (batch_size, 2)

        return outputs, attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)