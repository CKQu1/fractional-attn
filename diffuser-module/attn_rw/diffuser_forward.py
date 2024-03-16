import dgl.function as fn
import math
from dgl.nn.functional import edge_softmax
from time import time
from torch import nn
from tqdm import tqdm

from attn_utils import get_hidden_states
from models.utils import mask_attention_score

dataset_name = "imdb"
max_length = 1024

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
g.ndata['v'] =  value_vectors.reshape(-1, num_heads, head_dim) #BN,H,D

print("Graph random walk")
t0 = time()    

g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]
g.apply_edges(mask_attention_score)   #kq
e = g.edata.pop('score') 
g.edata['score'] = edge_softmax(g, e)
g.edata['score']= nn.functional.dropout(g.edata['score'], p=p_dropout, training=training_dropout)
g.ndata["h"] = g.ndata["v"]

# check what brute force method generates g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))

# get weight matrix
"""
src, dst = g.edges()
num_nodes, num_edges = g.num_nodes(), g.num_edges()   
wadj = torch.zeros([num_heads, num_nodes, num_nodes]) 
for eidx in tqdm(range(num_edges)):  # there should be a more efficient message passing method
    src_node, dst_node = src[eidx], dst[eidx]
    wadj[:, src_node, dst_node] = g.edata['score'][eidx].squeeze()

#h_new = (g.ndata["h"].T @ wadj.T).T
h_new = wadj.T @ g.ndata["h"]
g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))

h_new
g.ndata["h"]
"""

alpha = 0.1
for _ in tqdm(range(5)):
    g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))
    g.apply_nodes(lambda nodes: {'h' : (1.0 - alpha) * nodes.data['h'] + alpha * nodes.data['v']})
    g.ndata['h']= nn.functional.dropout(g.ndata['h'], p=p_dropout, training=training_dropout)

print(f"Graph RW: {time() - t0}s")

attn_output = g.ndata['h'] #BN,H,D
print(f"Diffuser attn_output computation: {time() - t0}s")

attn_output = attn_output.reshape(batch_size, seq_len,  num_heads, head_dim) # B,N,H,D        
assert attn_output.size() == (batch_size, seq_len, num_heads, head_dim), "Unexpected size"
attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
outputs = (attn_output.transpose(0, 1),) # Seq,B,D