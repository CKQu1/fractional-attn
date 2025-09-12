import torch
from torch import nn
from vit_models.model_utils import NewGELUActivation, PatchEmbeddings, Embeddings, MLP
from vit_models.att import MultiHeadAttention, RDFNSMultiHeadAttention, SPFNSMultiHeadAttention


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        # -------------------- DIFFERENT TYPES OF ATTN HERE --------------------
        if config['model_name'] == 'dpvit':
            self.attention = MultiHeadAttention(config)
        elif config['model_name'] == 'rdfnsvit':
            self.attention = RDFNSMultiHeadAttention(config)
        elif config['model_name'] == 'spfnsvit':
            self.attention = SPFNSMultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

        self.is_preln = config['is_preln']

    # Self-attention
    def forward(self, x, output_attentions=False):        
        # Norm & Add (Pre-LayerNorm)
        if self.is_preln:
            # Self-attention
            attention_output, attention_probs = \
                self.attention(self.layernorm_1(x), output_attentions=output_attentions)
            # Skip connection
            x = x + attention_output
            # Feed-forward network
            mlp_output = self.mlp(self.layernorm_2(x))
            # Skip connection
            x = x + mlp_output
        # Add & Norm (Post-LayerNorm)
        else:
            attention_output, attention_probs = \
                self.attention(x, output_attentions=output_attentions)    
            # Skip connection + layernorm
            x = self.layernorm_1(x + attention_output)                
            # Feed-forward network
            mlp_output = self.mlp(x)     
            # Skip connection + layernorm
            x = self.layernorm_2(x + mlp_output)           
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)  


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)