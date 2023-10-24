import dgl.function as fn
import math
import matplotlib.pyplot as plt
import os
import sys
import torch
from dgl.nn.functional import edge_softmax
from matplotlib.colors import ListedColormap
from os.path import join
from time import time
from torch import nn
from tqdm import tqdm

sys.path.append(os.getcwd())
from attn_rw.attn_utils import get_hidden_states
from models.utils import mask_attention_score
from path_setup import droot

cmp=ListedColormap(['white','black'])

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

#bool_mask = (attention_mask>=0 )
bool_mask = (attention_mask>0 )
g = g.local_var()
g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
g.ndata['q'] =  query_vectors.reshape(-1, num_heads, head_dim) #BN,H,D
g.ndata['k'] =  key_vectors.reshape(-1, num_heads, head_dim) #BN,H,D
g.ndata['v'] =  value_vectors.reshape(-1, num_heads, head_dim) #BN,H,D

print("Graph random walk")
t0 = time()    

g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]
dummy_score = g.edata['score']  # delete
g.apply_edges(mask_attention_score)   #kq
e = g.edata.pop('score') 
g.edata['score'] = edge_softmax(g, e)
#g.edata['score']= nn.functional.dropout(g.edata['score'], p=p_dropout, training=training_dropout)
g.ndata["h"] = g.ndata["v"]

# check what brute force method generates g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))

src, dst = g.edges()
num_nodes, num_edges = g.num_nodes(), g.num_edges()   
wadj = torch.zeros([num_heads, num_nodes, num_nodes]) 
for eidx in tqdm(range(num_edges)):  # there should be a more efficient message passing method
    src_node, dst_node = src[eidx], dst[eidx]
    #wadj[:, src_node, dst_node] = g.edata['score'][eidx].squeeze()
    wadj[:, src_node, dst_node] = torch.ones(2)

# check if masked correctly
#wadj[wadj!=0] = 1
nrows, ncols = 2, wadj.shape[0]
fig, axs = plt.subplots(nrows, ncols)
#axs = axs.flat
for hidx in range(wadj.shape[0]):
    # cmap=cmp
    axs[0,hidx].imshow(wadj[hidx].detach().numpy(), cmap="Greys", interpolation=None)
axs[1,0].imshow(wadj[0,:seq_len,:seq_len].detach().numpy(), cmap="Greys", interpolation=None)
axs[1,1].imshow(wadj[0,seq_len:2*seq_len,seq_len:2*seq_len].detach().numpy(), cmap="Greys", interpolation=None)

# masking across different heads are equal
print(f"{(wadj[0]==wadj[1]).sum()}, {(batch_size*seq_len)**2}")

# masking across different batches are equal
for i in range(batch_size-1):
    start, end = seq_len*(i + 1), seq_len*(i + 2)
    print(f"batch {i}")
    print(f"{(wadj[0,:seq_len,:seq_len] == wadj[0,start:end,start:end]).sum()}, {seq_len**2}")

#plt.show()
plt.savefig(join(droot, "sparse_pattern", "diffusion_mask.pdf"))