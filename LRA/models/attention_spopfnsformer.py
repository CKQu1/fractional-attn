import torch
import torch.nn as nn
import math

from torch.nn import functional as F

class SPOPFNSAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.alpha = config["alpha"]
        self.bandwidth = config["bandwidth"]     
        self.a = config["a"]   

        if self.alpha < 2:
            self.d_intrinsic = config["d_intrinsic"]
        self.sphere_radius = config["sphere_radius"]
        self.mask_val = config["mask_val"]       

    def forward(self, Q, K, V, mask):

        alpha, bandwidth, a = self.alpha, self.bandwidth, self.a
        d_intrinsic, sphere_radius, mask_val = self.d_intrinsic, self.sphere_radius, self.mask_val

        # geodesic distance on sphere
        eps = 1e-7  # for limiting the divergence from acos                     
        g_dist = torch.acos(torch.clamp(torch.matmul(Q, torch.transpose(K, -2, -1)), -1 + eps, 1 - eps)) * sphere_radius

        if mask is not None:        
            # type 1: key_pad_mask
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand([-1,self.num_head,1,-1])
            # print(f'mask_expanded shape: {mask_expanded.shape}')
            # print(mask_expanded)
            # quit()
            g_dist = g_dist.masked_fill(mask_expanded==0, mask_val)

        # Calculate the attention scores
        if alpha < 2:
            attn_score = (1 + g_dist / bandwidth**0.5) ** (-d_intrinsic - alpha)
        else:
            attn_score = torch.exp((-g_dist / bandwidth**0.5) ** (alpha / (alpha - 1)))

        if a == 0:
            # attn_score = attn_score.masked_fill_(attention_mask==0, -1e9) # Mask
            attn = F.normalize(
                attn_score, p=1, dim=3
            )  # can do this as the attn weights are always positive
        else:
            """
            D_inv_row = torch.diag_embed(
                attn_score.sum(-1) ** (-a)
            )  # inverse of degree matrix of attn_score
            D_inv_col = torch.diag_embed(
                attn_score.sum(-2) ** (-a)
            )  # inverse of degree matrix of attn_score
            K_tilde = D_inv_row @ attn_score @ D_inv_col
            # K_tilde = K_tilde.masked_fill_(attention_mask==0, -1e9) # Mask
            """
            N_R = attn_score.sum(-1)  # row sum
            N_C = attn_score.sum(-2)  # col sum                
            K_tilde = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2) 

            attn = F.normalize(
                K_tilde, p=1, dim=3
            )  # can do this as the attn weights are always positive

        attn = self.drop_attn(attn)
        #X = torch.matmul(attn, V)
        X = attn @ V

        return X

    def extra_repr(self):
        return f'spopfnsformer-alpha={self.alpha}-bandwidth={self.bandwidth}-a={self.a}'
