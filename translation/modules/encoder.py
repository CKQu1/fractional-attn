from torch import nn
from .interfaces import Module
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class EncoderLayer(Module):
    def __init__(self, config):
        super().__init__()

        self.d_model, self.num_heads, self.dropout_rate = config['d_model'], config['num_heads'], config['dropout_rate']

        self.self_attention = MultiHeadAttention(config)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.layer_norm1 = nn.LayerNorm(self.d_model)

        self.ffn = FeedForwardNetwork(self.d_model)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.layer_norm2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # Multi-headed attention and residual connection + layer norm
        # Dropout is applied to sub-layer output, before residual and norm
        attention_out = self.self_attention(queries=x, keys=x, values=x)
        x = self.layer_norm1(x + self.dropout1(attention_out))

        # Feed-forward network and another residual + layer norm
        ffn_out = self.ffn(x)
        return self.layer_norm2(x + self.dropout2(ffn_out))
