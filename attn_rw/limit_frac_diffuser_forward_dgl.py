import dgl
import dgl.function as fn
import dgl.ops as ops
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

# transformer model is fixed for now
query_vectors, key_vectors, value_vectors, attention_mask, g, dims_all, dropout_args = get_hidden_states(dataset_name, max_length)
seq_len, batch_size, num_heads, head_dim, embed_dim = dims_all
p_dropout, training_dropout = dropout_args

# normalize query
query_vectors /= math.sqrt(head_dim)

query_vectors = query_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) # (B,N,H,D)
key_vectors = key_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) # (B,N,H,D)
value_vectors = value_vectors.view(seq_len, batch_size, num_heads, head_dim).transpose(0, 1) #B,N,H,D

bool_mask = (attention_mask>=0 )
g = g.local_var()
g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
g.ndata['q'] =  query_vectors.reshape(-1, num_heads, head_dim) #BN,H,D
g.ndata['k'] =  key_vectors.reshape(-1, num_heads, head_dim) #BN,H,D
#g.ndata['v'] =  value_vectors.reshape(-1, num_heads, head_dim) #BN,H,D

print("Method 2")
t0 = time()

g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]    
g.edata['score'] = g.edata['score'].exp()  # exponential taken here, edge weights are non-neg
g.apply_edges(mask_attention_score)   #kq
# creating Bmat in the format of ndata and store in 'score'   
BN = batch_size * seq_len

rev = dgl.reverse(g)
out_degree = ops.copy_e_sum(rev, g.edata['score'])  # out deg (verified!)
rhos = torch.max(out_degree, axis=0).values    

# pass score into B and modify edge weights
g.edata['B'] = g.edata.pop('score')  # score is popped into B for saving space
diag_count = 0
src, dst = g.edges()
num_nodes, num_edges = g.num_nodes(), g.num_edges()    
for eidx in tqdm(range(num_edges)):  # there should be a more efficient message passing method
    src_node, dst_node = src[eidx], dst[eidx]
    if src_node == dst_node:
        g.edata['B'][eidx] += rhos - out_degree[diag_count]
        diag_count += 1    

# initialization of B^n and L^alpha
g.ndata['B_power'] = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1).transpose(0,1)  # stacked identity matrix tranposed
g.ndata['L_gamma'] = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1).transpose(0,1)  # for storing fractional Laplacian

t1 = time()
print(f"Computation done in {t1 - t0}s!")    

# trick for efficiently taking powers of Bmat (need to make sure the order of transpose is correct)    
N_approx = 6    
numerator, denominator = 1, 1
for ii in tqdm(range(1, N_approx+1)):
    numerator *= (gamma - ii + 1) * (-1)
    denominator *= ii * rhos
    coef = numerator/denominator            
    #print(coef)  # delete
    g.update_all(fn.u_mul_e('B_power', 'B', 'm'), fn.sum('m', 'B_power'))
    g.apply_nodes(lambda nodes: {'L_gamma' : nodes.data['L_gamma'] + coef * nodes.data['B_power']})
    #g.ndata['L_gamma']= nn.functional.dropout(g.ndata['L_gamma'], p=p_dropout, training=training_dropout)    
g.apply_nodes(lambda nodes: {'L_gamma' : rhos**gamma * nodes.data['L_gamma']})

# extract diagonals of L_gamma for limiting/stationary distribution
L_gamma_diags = torch.diagonal( g.ndata["L_gamma"].transpose(0,1), dim1=-1, dim2=-2 )
L_gamma_diags = L_gamma_diags / L_gamma_diags.sum(axis=1).unsqueeze(dim=-1)
s_dist = L_gamma_diags.unsqueeze(dim=-2).repeat(1,BN,1)
value_vectors = value_vectors.reshape(-1, num_heads, head_dim)
attn_output = torch.bmm(s_dist, value_vectors.transpose(0,1))

print(f"FracSelfAttn Method dgl: {time() - t0}s")

attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
outputs = (attn_output.transpose(0, 1),) # Seq,B,D    