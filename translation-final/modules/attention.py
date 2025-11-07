import torch
import math
from torch import nn
from .interfaces import Module
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

class MultiHeadAttention(Module):
    def __init__(self, config, use_mask=False):
        super().__init__()

        self.d_model = d_model = config['d_model']
        self.num_heads = num_heads = config['num_heads']
        self.is_op = config['is_op']
        self.qkv_bias = config['qkv_bias']        
        assert d_model % num_heads == 0, 'D_MODEL must be divisible by NUM_HEADS'
        self.use_mask = use_mask

        self.model_name = config['model_name']
        # FNS
        if self.model_name == 'rdfnsformer':
            self.alpha, self.bandwidth, self.a = config['alpha'], config['bandwidth'], config['a']
            self.is_rescale_dist = config['is_rescale_dist']
            d_k = self.d_model // self.num_heads
            if self.alpha < 2:
                self.d_intrinsic = d_k                
            if self.is_rescale_dist:    
                if self.alpha < 2:            
                    #self.dist_scale = d_k**0.5 / (2**(1/d_k) - 1)
                    #self.dist_scale = d_k**0.5 / (d_k**(1/d_k) - 1)
                    #self.dist_scale = (d_k**(1/d_k) - 1)
                    #self.dist_scale = 2*d_k**1.5
                    #self.dist_scale = d_k**1.5
                    self.dist_scale = d_k
                else:
                    self.dist_scale = math.sqrt(d_k)

        # w_q_i projects D_MODEL to D_MODEL / NUM_HEADS. However, there are
        # NUM_HEADS parallel attention layers that are concatenated, so in the
        # end output dim is still D_MODEL / NUM_HEADS * NUM_HEADS = D_MODEL
        if not self.is_op:
            self.w_q = nn.Linear(d_model, d_model, bias=self.qkv_bias)
            self.w_k = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        else:
            self.w_q = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))
            self.w_k = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))
        self.w_v = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.w_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        # queries, keys, values = (batch, seq, 512)
        # w_q = (512, 512)
        # queries @ w_q.t = (batch, seq, 512)
        # split_heads = (batch, 8, seq, 64)
        q = self.split_heads(self.w_q(queries))
        k = self.split_heads(self.w_k(keys))
        v = self.split_heads(self.w_v(values))

        # Perform NUM_HEADS parallel single-head attention
        if self.model_name == 'dpformer':
            attention = self.scaled_dot_product_attention(q, k, v)
        elif self.model_name == 'rdfnsformer':
            attention = self.rdfns_attention(q, k, v)

        # Concatenate and return multi-headed results
        # (batch, 8, seq, 64) -> (batch, seq, 512)
        merged = self.merge_heads(attention)

        # Apply final projection matrix
        return self.w_o(merged)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()

        # Split D_MODEL into NUM_HEADS channels of D_MODEL // NUM_HEADS each
        # Now shape is (batch, seq, num_heads, d_model/num_heads)
        heads = x.reshape(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads)

        # However, we want (batch, num_heads, seq, d_model/num_heads) because each tensor
        # of size (seq, d_model/num_heads) represents a single-head attention
        return heads.transpose(2, 1)

    def merge_heads(self, x):
        # Concatenate multi-headed results back into shape (batch, seq, d_model)
        # This is the inverse of split_heads
        batch_size, _, seq_len, _ = x.size()

        # Switch back to shape (batch, seq, num_heads, d_model)
        transposed = x.transpose(1, 2)

        # Merge last two dimensions
        return transposed.reshape(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, q, k, v):
        # Inputs are size (batch, num_heads, seq, d_model/num_heads)
        d_k = self.d_model // self.num_heads
        compatibility = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)

        """
        Use lower-triangular mask to prevent leftward information flow
        Fill upper triangle with negative infinity to zero out those values during softmax

        seq     weights      values          output
        0       [1 0 0]   [ --- a --- ]   [ a + 0 + 0 ]
        1       [1 1 0] * [ --- b --- ] = [ a + b + 0 ]
        2       [1 1 1]   [ --- c --- ]   [ a + b + c ]

        At seq=0, can only attend to seq=0
        At seq=1, can attend to both seq=0 and seq=1
        And so on...
        """
        if self.use_mask:
            seq_len = compatibility.size(-1)
            mask = torch.triu(  # Prevents leftward flow of information in target seq
                torch.ones(seq_len, seq_len, dtype=torch.bool, requires_grad=False),
                diagonal=1
            ).to(self.device)
            compatibility = torch.masked_fill(compatibility, mask, float('-inf'))

        # Apply softmax along the last dimension
        value_weights = self.softmax(compatibility)

        # print(f'q shape: {q.shape}')  # delete
        # print(f'k shape: {k.shape}')  # delete
        # print(f'v shape: {v.shape}')  # delete
        # print(f'compatibility shape: {compatibility.shape}')  # delete
        # print(f'value_weights shape: {value_weights.shape}')  # delete
        # quit()  # delete

        # Weight values by softmax results
        return torch.matmul(value_weights, v)
    
    def rdfns_attention(self, q, k, v):
        # Inputs are size (batch, num_heads, seq, d_model/num_heads)        
        g_dist = torch.cdist(q, k, p=2)
        if self.is_rescale_dist:
            g_dist = g_dist / self.dist_scale

        if self.alpha < 2:
            compatibility = (1 + g_dist/self.bandwidth**0.5)**(-self.d_intrinsic-self.alpha)
        else:
            compatibility = torch.exp(-(g_dist/self.bandwidth**0.5)**(self.alpha/(self.alpha-1)))

        # Same as above
        if self.use_mask:
            seq_len = compatibility.size(-1)
            mask = torch.triu(  # Prevents leftward flow of information in target seq
                torch.ones(seq_len, seq_len, dtype=torch.bool, requires_grad=False),
                diagonal=1
            ).to(self.device)
            # mask fractional attention score with 0
            compatibility = torch.masked_fill(compatibility, mask, 0)  

        if self.a > 0:  # this case is not executed since we always set a = 0
            N_R = compatibility.sum(-1)  # row sum
            N_C = compatibility.sum(-2)  # col sum
            K_tilde = (N_R**(-self.a)).unsqueeze(-1) * compatibility * (N_C**(-self.a)).unsqueeze(-2)

            value_weights = F.normalize(K_tilde,p=1,dim=3)  # can do this as the attn weights are always positive
        else:                      
            value_weights = F.normalize(compatibility,p=1,dim=3)  # can do this as the attn weights are always positive

        # print(f'q shape: {q.shape}')  # delete
        # print(f'k shape: {k.shape}')  # delete
        # print(f'v shape: {v.shape}')  # delete
        # print(f'compatibility shape: {compatibility.shape}')  # delete
        # print(f'value_weights shape: {value_weights.shape}')  # delete
        # quit()  # delete

        # Weight values by softmax results
        return torch.matmul(value_weights, v)    
