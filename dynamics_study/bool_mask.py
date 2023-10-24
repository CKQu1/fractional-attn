import dgl
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
from models.utils import mask_attention_score, frac_mask_attention_score
from path_setup import droot

cmp=ListedColormap(['white','black'])

dataset_name = "imdb"
max_length = 1024

# reset graph
seq_len, batch_size, num_heads, head_dim, embed_dim = 4,3,2,7,8
BN = batch_size * seq_len
src_dst = torch.ones([BN,BN]).nonzero()
src, dst = src_dst[:,0], src_dst[:,1]
g = dgl.graph((src, dst))

# reset attn mask
attention_mask = torch.ones([batch_size, seq_len])
attention_mask[:,2:] = 0
attention_mask[:,3] = 1

# -------------------- Method 1: Use DGL --------------------

#bool_mask = (attention_mask>=0 )
bool_mask = (attention_mask>0 )
g = g.local_var()
g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)  #BN,1

print("Graph random walk")
t0 = time()    

#g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]
wadj_og = torch.ones(num_heads, BN, BN)
dummy_score = torch.ones((BN**2, num_heads, 1))  # delete
g.edata['score'] = dummy_score
print(f"dummy_score: {dummy_score.sum()}")
#g.apply_edges(mask_attention_score)   #kq
g.apply_edges(frac_mask_attention_score)
e = g.edata['score']
#e = g.edata.pop('score') 
#g.edata['score'] = edge_softmax(g, e)
#g.edata['score']= nn.functional.dropout(g.edata['score'], p=p_dropout, training=training_dropout)

# this is assuming the edges have been established
"""
src, dst = g.edges()
num_nodes, num_edges = g.num_nodes(), g.num_edges()   
wadj = torch.zeros([num_heads, num_nodes, num_nodes]) 
for eidx in tqdm(range(num_edges)):  # there should be a more efficient message passing method
    src_node, dst_node = src[eidx], dst[eidx]
    #wadj[:, src_node, dst_node] = g.edata['score'][eidx].squeeze()
    wadj[:, src_node, dst_node] = torch.ones(2)
"""

num_nodes, num_edges = g.num_nodes(), g.num_edges()   
wadj = torch.zeros([num_heads, num_nodes, num_nodes]) 
e_remain = e.squeeze().nonzero()

num_edges = g.num_edges()
src, dst = g.edges()
for eidx in tqdm(range(e_remain.shape[0])):  # there should be a more efficient message passing method
    src_node, dst_node = src[e_remain[eidx][0]], dst[e_remain[eidx][0]]
    wadj[:, src_node, dst_node] = torch.ones(2)

# -------------------- Method 2: Torch only --------------------

wadj2 = torch.ones([num_heads, num_nodes, num_nodes]) 
bool_mask2 = (attention_mask>0).reshape(-1).unsqueeze(-1)
wadj2.masked_fill_((bool_mask2.int()@bool_mask2.int().T)==False, 0)  # attention mask

# check if masked correctly
nrows, ncols = 2, wadj.shape[0]
fig, axs = plt.subplots(nrows, ncols)
for hidx in range(wadj.shape[0]):
    # cmap=cmp
    axs[0,hidx].imshow(wadj[hidx].detach().numpy(), cmap="Greys", interpolation=None)
axs[1,0].imshow(wadj2[0,:seq_len,:seq_len].detach().numpy(), cmap="Greys", interpolation=None)
axs[1,1].imshow(wadj2[0,seq_len:2*seq_len,seq_len:2*seq_len].detach().numpy(), cmap="Greys", interpolation=None)

# masking across different heads are equal
print(f"{(wadj[0]==wadj[1]).sum()}, {(batch_size*seq_len)**2}")

# different ways of masking are equal
print(f"{(wadj[0]==wadj2[1]).sum()}, {(batch_size*seq_len)**2}")

# masking across different batches are equal
for i in range(batch_size-1):
    start, end = seq_len*(i + 1), seq_len*(i + 2)
    print(f"batch {i}")
    print(f"{(wadj[0,:seq_len,:seq_len] == wadj[0,start:end,start:end]).sum()}, {seq_len**2}")

#plt.show()
plt.savefig(join(droot, "sparse_pattern", "bool_mask.pdf"))