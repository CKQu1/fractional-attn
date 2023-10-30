import torch
import math
import dgl.function as fn
import dgl.ops as ops
import numpy as np
import operator as operator
from functools import reduce

from dgl import reverse
from dgl.nn.functional import edge_softmax
from dgl.ops import copy_e_sum
from models.diffuser_utils import *
from models.utils import *
from torch import nn

#def frac_C(n, k):
#    return reduce(op.mul, np.arange(n, n-k, -1), 1) / reduce(op.mul, range(1, k+1, 1), 1)

class DiffuserSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DiffuserAttention(nn.Module):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        assert isinstance(kwargs.get('with_frac'), bool), "with_frac must be boolean"
        self.with_frac = kwargs.get('with_frac')
        if self.with_frac:
            error_message = f"gamma = {kwargs.get('gamma')} with type {type(kwargs.get('gamma'))} is ill-defined!"
            assert 0 < kwargs.get('gamma') < 1, error_message
            gamma = kwargs.get('gamma')
            self.self = SymLimitFracSelfAttention(config, layer_id, gamma)
            #self.self = LimitFracSelfAttention(config, layer_id, gamma)
            #self.self = LimitFracSelfAttention_dgl(config, layer_id, gamma)
        else:
            self.self = DiffuserSelfAttention(config, layer_id)          
        self.output = DiffuserSelfOutput(config)

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.self(
            hidden_states,
            g=g,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


class DiffuserSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D
        
        bool_mask = (attention_mask>=0 )
        g = g.local_var()
        g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
        g.ndata['q'] =  query_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['k'] =  key_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['v'] =  value_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]
        g.apply_edges(mask_attention_score)   #kq
        e = g.edata.pop('score') 
        g.edata['score'] = edge_softmax(g, e)
        g.edata['score']= nn.functional.dropout(g.edata['score'], p=self.dropout, training=self.training)
        
        g.ndata["h"] = g.ndata["v"]
        alpha = 0.1
        for _ in range(5):
            g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))
            g.apply_nodes(lambda nodes: {'h' : (1.0 - alpha) * nodes.data['h'] + alpha * nodes.data['v']})
            g.ndata['h']= nn.functional.dropout(g.ndata['h'], p=self.dropout, training=self.training)

        attn_output = g.ndata['h'] #BN,H,D
        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs

# Fractional random walk (normalized fractional Laplacian applied)
"""
class DiffuserFracSelfAttention(nn.Module):
    def __init__(self, config, layer_id, gamma):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.gamma = gamma

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D
        
        bool_mask = (attention_mask>=0 )
        g = g.local_var()
        g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
        g.ndata['q'] =  query_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['k'] =  key_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['v'] =  value_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        
        # ----- Issue 1: order of dropout -----
        # option 1: apply nn.functional.dropout to QK
        # option 2: apply nn.functional.dropout to W
        # get Laplacian and B matrix
        # weight matrix (this doesn't incorporate the weights generated from QK)

        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))
        g.edata['score'] = g.edata['score'].exp()   # exponential taken here, representing the true edge weight which are positive
        g.apply_edges(mask_attention_score)   #kq
        
        #g.edata['out_deg'] = g.edata['weight']
        #g.update_all(fn.copy_e('out_deg', 'm'), fn.sum('m', 'out_deg'))

        #in_degree = copy_e_sum(g, g.edata['score'])    # in deg
        rev = reverse(g)
        out_degree = copy_e_sum(rev, g.edata['score'])  # out deg (verified!)
        rhos = torch.max(out_degree, axis=0).values

        # somehow all nodes are self-connected, this needs to be double-checked
        edge_shape = list(g.edata['score'].shape)
        num_nodes, num_edges = g.num_nodes(), g.num_edges()
        Bmat = torch.zeros([num_nodes, num_nodes] + edge_shape[1:]) # requires_grad=True
                
        # get weight adjacency matrix
        #src, dst = g.edges()    
        #for eidx in range(num_edges):
        #    src_node, dst_node = src[eidx], dst[eidx]
        #    Bmat[src_node, dst_node,:] = g.edata['score'][eidx]          
        #quit()            
        
        with torch.no_grad():
            diag_count = 0
            src, dst = g.edges()    
            for eidx in range(num_edges):
                src_node, dst_node = src[eidx], dst[eidx]
                if src_node == dst_node:
                    Bmat[src_node, dst_node,:] = g.edata['score'][eidx].clone() - rhos.clone() - out_degree[diag_count].clone()
                    diag_count += 1
                else:
                    Bmat[src_node, dst_node,:] = g.edata['score'][eidx].clone()
            
        #N_approx = 10   # probably as large as it can be, any larger will result in numerical degeneration
        N_approx = 6
        L_gamma = torch.eye(num_nodes).reshape([num_nodes,num_nodes] + [1]*(len(edge_shape) - 1))
        L_gamma = L_gamma.repeat([1,1] +  edge_shape[1:])
        L_gamma.requires_grad = True

        Bmat_power = Bmat.clone()
        with torch.no_grad():
            numerator, denominator = 1, 1
            for ii in range(1, N_approx+1):
                numerator *= (self.gamma - ii + 1) * (-1)
                denominator *= ii * rhos
                coef = numerator/denominator        
                for head_idx in range(edge_shape[1]):
                    L_gamma = L_gamma.clone() + coef * Bmat_power
                    Bmat_power[:,:,head_idx] = (Bmat_power[:,:,head_idx].clone().squeeze() @ Bmat[:,:,head_idx].squeeze()).unsqueeze(Bmat.ndim - 2)       

            L_gamma *= rhos**self.gamma           
        # normalized version
        L_gamma_normalized = L_gamma.clone()
        with torch.no_grad():
            for head_idx in range(edge_shape[1]):
                L_gamma_normalized[:,:,head_idx] = (torch.diag( 1/torch.diag(L_gamma_normalized[:,:,head_idx].clone().squeeze()) ) @ L_gamma_normalized[:,:,head_idx].clone().squeeze()).unsqueeze(Bmat.ndim - 2)
            
        # applying dropout in a similar fashion as DiffuserSelfAttention()
        L_gamma_normalized = nn.functional.dropout(L_gamma_normalized, p=self.dropout, training=self.training)

        # for checking whether the normalized version has rows summed up to zero        
        #for head_idx in range(edge_shape[1]):
        #    print(L_gamma_normalized[:,:,head_idx].sum(1))
        
        # realization of discrete time fractional RW
        RW_steps = 5
        attn_output = g.ndata['v'] #BN,H,D
        for _ in range(RW_steps):
            for head_idx in range(edge_shape[1]):        
                attn_output[:,head_idx] = L_gamma_normalized[:,:,head_idx].squeeze() @ attn_output[:,head_idx].clone()
                # add dropout for training
                attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)

        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs
"""


# Limiting fractional diffusion (which only works for symmetrized weights)
class SymLimitFracSelfAttention(nn.Module):
    def __init__(self, config, layer_id, gamma):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.gamma = gamma

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D
        
        query_vectors =  query_vectors.reshape(-1, self.num_heads, self.head_dim)  #BN,H,D
        key_vectors =  key_vectors.reshape(-1, self.num_heads, self.head_dim)  #BN,H,D
        value_vectors =  value_vectors.reshape(-1, self.num_heads, self.head_dim)  #BN,H,D        
        
        Bmat = torch.bmm( key_vectors.transpose(0,1), query_vectors.transpose(0,1).transpose(1,2) )  # KQ product: H,BN,BN
        Bmat = Bmat.exp()  # no softmax applied, consistent with Diffuser RW model
        #Bmat = nn.functional.dropout(Bmat, p=self.dropout, training=self.training)

        # attn sparsification
        Bmat.masked_fill_((g>0)==False, 0)
        # alternative brute force way
        #with torch.no_grad():
        #   Bmat *= g

        # Bmat --> W
        bool_mask = (attention_mask>=0).reshape(-1).unsqueeze(-1).int()
        Bmat.masked_fill_((bool_mask @ bool_mask.T)==False, 0)  # attention mask
        # ----- ADDED SYMMETRIZATION (after masking) ------
        with torch.no_grad():
            Bmat += Bmat.clone().transpose(1,2)
            Bmat *= 0.5
        # --------------------------------
        # Bmat --> Bmat
        BN = seq_len * batch_size    
        out_degree = Bmat.sum(axis=2)
        rhos = out_degree.max(axis=1).values 
        with torch.no_grad():   
            Bmat -= torch.diag_embed(out_degree)
            Bmat += torch.diag_embed(rhos.unsqueeze(-1).repeat(1, BN))  # apply dropout after this?
        B_power = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(self.num_heads, 1, 1)
        L_gamma = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(self.num_heads, 1, 1)        

        Bmat = Bmat.to_sparse()  # convert to sparse for speed-up
        N_approx = 6
        numerator, denominator = 1, 1
        #error_tolerence = 1e-5  # should this be introduced?
        with torch.no_grad():
            for ii in range(1, N_approx+1):
                numerator *= (self.gamma - ii + 1) * (-1)
                denominator *= ii * rhos
                coef = numerator/denominator            
                B_power = torch.bmm(Bmat, B_power)  # bmm supports format sparse bmm dense only
                L_gamma += coef.unsqueeze(-1).unsqueeze(-1) * B_power
            L_gamma *= rhos.unsqueeze(-1).unsqueeze(-1)**self.gamma

        L_gamma_diags = torch.diagonal( L_gamma, dim1=-2, dim2=-1 )
        L_gamma_diags = L_gamma_diags / L_gamma_diags.sum(axis=1).unsqueeze(dim=-1)  # apply dropout after this?     
        s_dist = L_gamma_diags.unsqueeze(dim=-2).repeat(1,BN,1)  
        #s_dist = nn.functional.dropout(s_dist, p=self.dropout, training=self.training)   
        attn_output = torch.bmm(s_dist, value_vectors.transpose(0,1))
        #attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)

        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs


# Limiting fractional diffusion with dgl (the same as above)
class LimitFracSelfAttention_dgl(nn.Module):
    def __init__(self, config, layer_id, gamma):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.gamma = gamma

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}" 

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D

        bool_mask = (attention_mask>=0 )
        g = g.local_var()
        g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
        g.ndata['q'] =  query_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['k'] =  key_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        #g.ndata['v'] =  value_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D

        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]    
        g.edata['score'] = g.edata['score'].exp()  # exponential taken here, edge weights are non-neg
        g.apply_edges(frac_mask_attention_score)   #kq
        # creating Bmat in the format of ndata and store in 'score'   
        BN = batch_size * seq_len

        rev = reverse(g)
        out_degree = ops.copy_e_sum(rev, g.edata['score'])  # out deg (verified!)
        rhos = torch.max(out_degree, axis=0).values    

        # pass score into B and modify edge weights
        g.edata['B'] = g.edata.pop('score')  # score is popped into B for saving space
        diag_count = 0
        src, dst = g.edges()
        num_nodes, num_edges = g.num_nodes(), g.num_edges()    
        for eidx in range(num_edges):  # there should be a more efficient message passing method
            src_node, dst_node = src[eidx], dst[eidx]
            if src_node == dst_node:
                g.edata['B'][eidx] += rhos - out_degree[diag_count]
                diag_count += 1    

        # initialization of B^n and L^alpha
        g.ndata['B_power'] = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(self.num_heads, 1, 1).transpose(0,1)  # stacked identity matrix tranposed
        g.ndata['L_gamma'] = torch.eye(BN, requires_grad=True).reshape(1, BN, BN).repeat(self.num_heads, 1, 1).transpose(0,1)  # for storing fractional Laplacian

        # trick for efficiently taking powers of Bmat (need to make sure the order of transpose is correct)    
        N_approx = 6    
        numerator, denominator = 1, 1
        for ii in range(1, N_approx+1):
            numerator *= (self.gamma - ii + 1) * (-1)
            denominator *= ii * rhos
            coef = numerator/denominator            
            #print(coef)  # delete
            g.update_all(fn.u_mul_e('B_power', 'B', 'm'), fn.sum('m', 'B_power'))
            g.apply_nodes(lambda nodes: {'L_gamma' : nodes.data['L_gamma'] + coef * nodes.data['B_power']})
            #g.ndata['L_gamma']= nn.functional.dropout(g.ndata['L_gamma'], p=p_dropout, training=training_dropout)    
        g.apply_nodes(lambda nodes: {'L_gamma' : rhos**self.gamma * nodes.data['L_gamma']})

        L_gamma_diags = torch.diagonal( g.ndata['L_gamma'].transpose(0,1), dim1=-2, dim2=-1 )
        L_gamma_diags = L_gamma_diags / L_gamma_diags.sum(axis=1).unsqueeze(dim=-1)  # apply dropout after this?     
        s_dist = L_gamma_diags.unsqueeze(dim=-2).repeat(1,BN,1)  
        #s_dist = nn.functional.dropout(s_dist, p=self.dropout, training=self.training)   
        value_vectors = value_vectors.reshape(-1, self.num_heads, self.head_dim)
        attn_output = torch.bmm(s_dist, value_vectors.transpose(0,1))

        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs


# Fractional diffusion using dgl api
"""
class DiffuserFracSelfAttention_dgl(nn.Module):
    def __init__(self, config, layer_id, gamma):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.gamma = gamma

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

    def forward(
        self,
        hidden_states,
        g=None,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D
        
        bool_mask = (attention_mask>=0 )
        g = g.local_var()
        g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
        g.ndata['q'] =  query_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['k'] =  key_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['v'] =  value_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        
        # ----- Issue 1: order of dropout -----
        # option 1: apply nn.functional.dropout to QK
        # option 2: apply nn.functional.dropout to W
        # get Laplacian and B matrix
        # weight matrix (this doesn't incorporate the weights generated from QK)

        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))
        g.edata['score'] = g.edata['score'].exp()   # exponential taken here, representing the true edge weight which are positive
        g.apply_edges(frac_mask_attention_score)   #kq
        
        rev = reverse(g)
        out_degree = copy_e_sum(rev, g.edata['score'])  # out deg (verified!)
        rhos = torch.max(out_degree, axis=0).values

        # somehow all nodes are self-connected, this needs to be double-checked
        edge_shape = list(g.edata['score'].shape)
        num_nodes, num_edges = g.num_nodes(), g.num_edges()
        Bmat = torch.zeros([num_nodes, num_nodes] + edge_shape[1:]) # requires_grad=True                        
        
        with torch.no_grad():
            diag_count = 0
            src, dst = g.edges()    
            for eidx in range(num_edges):
                src_node, dst_node = src[eidx], dst[eidx]
                if src_node == dst_node:
                    Bmat[src_node, dst_node,:] = g.edata['score'][eidx].clone() - rhos.clone() - out_degree[diag_count].clone()
                    diag_count += 1
                else:
                    Bmat[src_node, dst_node,:] = g.edata['score'][eidx].clone()
            
        #N_approx = 10   # probably as large as it can be, any larger will result in numerical degeneration
        N_approx = 6
        L_gamma = torch.eye(num_nodes).reshape([num_nodes,num_nodes] + [1]*(len(edge_shape) - 1))
        L_gamma = L_gamma.repeat([1,1] +  edge_shape[1:])
        L_gamma.requires_grad = True

        Bmat_power = Bmat.clone()
        with torch.no_grad():
            numerator, denominator = 1, 1
            for ii in range(1, N_approx+1):
                numerator *= (self.gamma - ii + 1) * (-1)
                denominator *= ii * rhos
                coef = numerator/denominator        
                for head_idx in range(edge_shape[1]):
                    L_gamma = L_gamma.clone() + coef * Bmat_power
                    Bmat_power[:,:,head_idx] = (Bmat_power[:,:,head_idx].clone().squeeze() @ Bmat[:,:,head_idx].squeeze()).unsqueeze(Bmat.ndim - 2)       

            L_gamma *= rhos**self.gamma           
    
        # applying dropout in a similar fashion as DiffuserSelfAttention()
        L_gamma = nn.functional.dropout(L_gamma, p=self.dropout, training=self.training)
        
        # realization of discrete time fractional RW
        RW_steps = 5
        attn_output = g.ndata['v'] #BN,H,D
        for _ in range(RW_steps):
            for head_idx in range(edge_shape[1]):        
                attn_output[:,head_idx] = L_gamma[:,:,head_idx].squeeze() @ attn_output[:,head_idx].clone()
                # add dropout for training
                attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)

        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = (attn_output.transpose(0, 1),) # Seq,B,D

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs
"""        