import math
import torch
from time import time
from torch import nn
from tqdm import tqdm

from attn_utils import get_hidden_states
from models.utils import mask_attention_score

dataset_name = "imdb"
max_length = 1024
gamma = 0.5
use_dgl = False

# transformer model is fixed for now
query_vectors, key_vectors, value_vectors, attention_mask, g, dims_all, dropout_args = get_hidden_states(dataset_name, max_length=max_length, use_dgl=use_dgl)
seq_len, batch_size, num_heads, head_dim, embed_dim = dims_all
p_dropout, training_dropout = dropout_args

# normalize query
query_vectors /= math.sqrt(head_dim)

query_vectors = query_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) # (B,N,H,D)
key_vectors = key_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) # (B,N,H,D)
value_vectors = value_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) #B,N,H,D

query_vectors =  query_vectors.reshape(-1, num_heads, head_dim)  #BN,H,D
key_vectors =  key_vectors.reshape(-1, num_heads, head_dim)  #BN,H,D
value_vectors =  value_vectors.reshape(-1, num_heads, head_dim)  #BN,H,D      

t0 = time()

Bmat = torch.bmm( key_vectors.transpose(0,1), query_vectors.transpose(0,1).transpose(1,2) )  # KQ product: H,BN,BN
Bmat = Bmat.exp()  # no softmax applied, consistent with Diffuser

# attn sparsification
Bmat.masked_fill_((g>0)==False, 0)
# alternative brute force way
#with torch.no_grad():
#   Bmat *= g

bool_mask = (attention_mask>=0).reshape(-1).unsqueeze(-1)
Bmat.masked_fill_((bool_mask @ bool_mask.T)==False, 0)  # attention mask
BN = batch_size * seq_len    
out_degree = Bmat.sum(axis=2)
rhos = out_degree.max(axis=1).values    
Bmat -= torch.diag_embed(out_degree)
Bmat += torch.diag_embed(rhos.unsqueeze(-1).repeat(1, BN))
B_power = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1)
L_gamma = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1)  # row sum zero in the limit

N_approx = 6    
numerator, denominator = 1, 1
with torch.no_grad():
    for ii in tqdm(range(1, N_approx+1)):
        numerator *= (gamma - ii + 1) * (-1)
        denominator *= ii * rhos
        coef = numerator/denominator            
        B_power = torch.bmm(B_power, Bmat)  # apply dropout after this?
        L_gamma += coef.unsqueeze(-1).unsqueeze(-1) * B_power
    L_gamma *= rhos.unsqueeze(-1).unsqueeze(-1)**gamma

t1 = time()
print(f"Computation done in {t1 - t0}s!")

L_gamma_diags = torch.diagonal( L_gamma, dim1=-2, dim2=-1 )
L_gamma_diags = L_gamma_diags / L_gamma_diags.sum(axis=1).unsqueeze(dim=-1)  # apply dropout after this?
s_dist = L_gamma_diags.unsqueeze(dim=-2).repeat(1,BN,1)
attn_output = torch.bmm(s_dist, value_vectors.transpose(0,1))

print(f"FracSelfAttn Method without dgl: {time() - t0}s")

attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
outputs = (attn_output.transpose(0, 1),) # Seq,B,D   