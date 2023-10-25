import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
from os.path import join
from time import time
from torch import nn
from tqdm import tqdm

sys.path.append(os.getcwd())
from attn_rw.attn_utils import get_hidden_states
from models.utils import mask_attention_score
from path_setup import droot

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

# -------------------- Sparse pattern --------------------

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

#bool_mask = (attention_mask>=0).reshape(-1).unsqueeze(-1)
#Bmat.masked_fill_(bool_mask==False,0)  # attention mask

# check if masked correctly
Bmat[Bmat!=0] = 1
nrows, ncols = 1, Bmat.shape[0]
fig, axs = plt.subplots(nrows, ncols)
axs = axs.flat
for hidx in range(Bmat.shape[0]):
    axs[hidx].imshow(Bmat[hidx].detach().numpy(), cmap="Greys")

# save sparse pattern
plt.savefig(join(droot, "sparse_pattern", "Bmat_mask.pdf"))
plt.clf()

# -------------------- Fractional Laplacian convergence --------------------

BN = batch_size * seq_len    
out_degree = Bmat.sum(axis=2)
rhos = out_degree.max(axis=1).values    
Bmat -= torch.diag_embed(out_degree)
Bmat += torch.diag_embed(rhos.unsqueeze(-1).repeat(1, BN))
B_power = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1)
L_gamma = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1)  # row sum zero in the limit

N_approx = 8
diffs = []
error_types = ["Diagonal", "Max", "Frobenius"]
errors = torch.zeros([len(error_types), num_heads, N_approx])    
numerator, denominator = 1, 1
L_gamma_cur = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(num_heads, 1, 1)
with torch.no_grad():
    for ii in tqdm(range(1, N_approx+1)):
        numerator *= (gamma - ii + 1) * (-1)
        denominator *= ii * rhos
        coef = numerator/denominator            
        B_power = torch.bmm(B_power, Bmat)  # apply dropout after this?
        L_gamma += coef.unsqueeze(-1).unsqueeze(-1) * B_power
        #diffs.append( torch.diagonal(rhos.unsqueeze(-1).unsqueeze(-1)**gamma*(L_gamma-L_gamma_cur), dim1=-2, dim2=-1 ) )
        diff = rhos.unsqueeze(-1).unsqueeze(-1)**gamma*(L_gamma - L_gamma_cur)
        for hidx in range(num_heads):
            # trace squared difference
            errors[0, hidx, ii-1] = (torch.diagonal(diff,dim1=-2,dim2=-1)[hidx]**2).sum()
            # difference in max difference
            errors[1, hidx, ii-1] = diff[hidx].max()            
            # difference in Frobenius norm
            errors[2, hidx, ii-1] = diff[hidx].sum()**2
        L_gamma_cur = L_gamma
    L_gamma *= rhos.unsqueeze(-1).unsqueeze(-1)**gamma

nrows, ncols = 2, 2
fig, axs = plt.subplots(nrows, ncols)
axs = axs.flat
alphas = np.linspace(0.3, 1, num_heads)[::-1]

for idx, error_type in enumerate(error_types):
    axis = axs[idx]
    for hidx in range(num_heads):
        axis.plot( list(range(1, N_approx+1)), errors[idx,hidx,:], alpha=alphas[hidx], label=f"Head {hidx+1}" )
    axis.set_title(f"{error_type}")
axs[0].legend()

fig_path = join(droot, "laplacian_dynamics")
if not os.path.isdir(fig_path): os.makedirs(fig_path)
plt.savefig(join(fig_path, "lapl_convergence.pdf"))

t1 = time()
print(f"Computation done in {t1 - t0}s!")