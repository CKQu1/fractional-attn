import torch
import torch.nn as nn
import math

from models.sinkhorn import SinkhornDistance

class SINKAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.mask_val = config["mask_val"]

        self.n_it = config["n_it"]
        self.bandwidth = config["bandwidth"]

    def forward(self, Q, K, V, mask):

        n_it = self.n_it
        bandwidth = self.bandwidth
        mask_val = self.mask_val

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)        

        if mask is not None:

            # type 1: key_pad_mask
            mask = mask.type(torch.bool)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_head,1,-1])
            dot.masked_fill_(mask_expanded, mask_val)

            # type 2:
            # dot = dot - 1e6 * (1 - mask[:, None, None, :])

        dot_shape = dot.shape

        attn_weights_soft = nn.Softmax(dim=-1)(dot)
        dot = dot.view(-1, dot_shape[2], dot_shape[3])  # (B,H,N,D)
        sink = SinkhornDistance(self.bandwidth, max_iter=n_it)
        attn = sink(dot)[0]

        attn = attn * attn.shape[-1]
        attn = attn.view(dot_shape)  # (B,H,N,D)    

        attn = self.drop_attn(attn)        
        X = attn @ V

        return X

    def extra_repr(self):
        return f'sinkformer-n_it={self.n_it}'