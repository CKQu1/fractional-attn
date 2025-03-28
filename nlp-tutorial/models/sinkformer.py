import numpy as np
import torch
import torch.nn as nn
from models.sinkhorn import SinkhornDistance
from torch.nn.utils.parametrizations import orthogonal

from .model_utils import PositionWiseFeedForwardNetwork

class SINKSelfAttention(nn.Module):
    def __init__(self, config):
        super(SINKSelfAttention, self).__init__()
        #self.d_k = d_k
        self.head_dim = config['head_dim']
        self.n_it = config['n_it']

    def forward(self, q, k, v, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        
        n_it = self.n_it

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        attn_score.masked_fill_(attn_mask, -1e9)
        attn_score_shape = attn_score.shape
        attn_weights_soft = nn.Softmax(dim=-1)(attn_score)
        # |attn_score| : (batch_size, n_heads, q_len, k_len)
        attn_score = attn_score.view(-1, attn_score_shape[2], attn_score_shape[3])
        sink = SinkhornDistance(1, max_iter=n_it)
        attn_weights = sink(attn_score)[0]

        attn_weights = attn_weights * attn_weights.shape[-1]

        attn_weights = attn_weights.view(attn_score_shape)
        #
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        
        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)
        
        return output, attn_weights

class SINKAttention(nn.Module):
    #def __init__(self, d_model, n_heads, n_it=1, print_attention=False):
    def __init__(self, config):
        super(SINKAttention, self).__init__()
        self.d_model = d_model = config['d_model']
        self.n_heads = n_heads = config['n_heads']        
        #self.d_k = self.d_v = d_model//n_heads
        self.head_dim = head_dim = config['head_dim']

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']

        self.n_it = n_it = config['n_it']

        self.WQ = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        if not self.qk_share:
            self.WK = nn.Linear(d_model, d_model, bias=self.qkv_bias)

        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.scaled_dot_product_attn = SINKSelfAttention(config)
        self.linear = nn.Linear(n_heads * head_dim, d_model)
        #self.print_attention = print_attention
        
    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        
        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)          
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        if not self.qk_share:
            k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) 
            attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            attn, attn_weights = self.scaled_dot_product_attn(q_heads, q_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        # if self.print_attention:
        #     print('sum over columns of the attention matrix: ', attn_weights.sum(-2))
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)

        return output, attn_weights

class OPSINKAttention(nn.Module):
    def __init__(self, config):
        super(OPSINKAttention, self).__init__()         
        self.n_heads = n_heads = config['n_heads']
        self.d_model = d_model = config['d_model']
        #self.d_k = self.d_v = config['d_model']//self.n_heads
        self.head_dim = head_dim = config['head_dim']

        self.qkv_bias = config['qkv_bias']
        self.qk_share = config['qk_share']
        
        if not self.qk_share:
            self.WK = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))
        if self.n_heads > 1:
            self.WQ = orthogonal(nn.Linear(d_model, d_model, bias=self.qkv_bias))

        self.WV = nn.Linear(d_model, d_model, bias=self.qkv_bias)
        self.scaled_dot_product_attn = SINKSelfAttention(config)
        self.linear = nn.Linear(n_heads * head_dim, d_model)
        
    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        
        if self.n_heads > 1:
            q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q_heads = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        if not self.qk_share: 
            k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)         
            attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        else:
            attn, attn_weights = self.scaled_dot_product_attn(q_heads, q_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)

        return output, attn_weights              

class EncoderLayer(nn.Module):
    #def __init__(self, d_model, n_heads, p_drop, d_ff, n_it=1, print_attention=False):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model = config['d_model']
        self.p_drop = p_drop = config['p_drop']
        self.d_ff = d_ff = config['d_ff']        

        if not config['is_op']:
            self.mha = SINKAttention(config)
        else:
            self.mha = OPSINKAttention(config)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        return ffn_outputs, attn_weights

class SINKformer(nn.Module):
    """SINKformer is a stack of N encoder layers.

    Args:
        vocab_size (int)    : vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len    (int)    : input sequence length
        d_model    (int)    : number of expected features in the input
        n_layers   (int)    : number of sub-encoder-layers in the encoder
        n_heads    (int)    : number of heads in the SINKAttention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model
        pad_id     (int)    : pad token id

    Examples:
    >>> encoder = SINKformer(vocab_size=1000, seq_len=512)
    >>> inp = torch.arange(512).repeat(2, 1)
    >>> encoder(inp)
    """
    
    def __init__(self, config):
        super(SINKformer, self).__init__()

        self.vocab_size = config['vocab_size']
        self.seq_len = config['seq_len']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.p_drop = config['p_drop']
        self.d_ff = config['d_ff']
        self.pad_id = config['pad_id']
        self.fix_embed = config['fix_embed']
        self.max_len = config['seq_len']

        # embeddings
        if not self.fix_embed:
            self.sinusoid_table = self.get_sinusoid_table(self.seq_len+1, self.d_model) # (seq_len+1, d_model)
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
            self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        else:
            #self.embeddings = ModelEmbeddings(config)

            from v2_models.model_utils import load_embeddings
            self.embeddings = load_embeddings(config)
            #self.embeddings.requires_grad_(False)
            for param in self.embeddings.parameters():
                param.requires_grad = False   

        # layers
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(self.n_layers)])
        # layers to classify
        self.linear = nn.Linear(self.d_model, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        if not self.fix_embed:
            outputs = self.embedding(inputs) + self.pos_embedding(positions)
            # |outputs| : (batch_size, seq_len, d_model)
        else:
            #outputs = self.embeddings(input_ids=inputs, position_ids=positions)
            outputs = self.embeddings(input_ids=inputs)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        attention_weights = []

        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attention_weights.append(attn_weights)
        
        outputs, _ = torch.max(outputs, dim=1)
        # |outputs| : (batch_size, d_model)
        outputs = self.softmax(self.linear(outputs))
        # |outputs| : (batch_size, 2)

        return outputs, attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)