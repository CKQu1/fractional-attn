import math
import torch
from torch import nn
from torch.nn import functional as F

from models.fns_att import FNSEncoder
from models.opfns_att import OPFNSEncoder
from models.dp_att import DPEncoder
from models.sink_att import SINKEncoder

class ClassifierHead(nn.Module):
    """Classifier head."""
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        if self.pooling_mode == 'FLATTEN':
            in_dim = config["hidden_size"] * config["max_length"]
        else:
            in_dim = config["hidden_size"]
        self.linear1 = nn.Linear(in_features=in_dim, out_features=config["intermediate_size"])
        self.relu = nn.ReLU() # CAN CHANGE THIS
        self.linear2 = nn.Linear(in_features=config["intermediate_size"], out_features=config["num_classes"])
        
    def forward(self, encoded, attention_mask=None):
        # Pooling across sequence length; B x L x H -> B x H
        if self.pooling_mode == 'MEAN':
            encoded = torch.mean(encoded*attention_mask.squeeze().unsqueeze(-1), dim=1)
        elif self.pooling_mode == 'SUM':
            encoded = torch.sum(encoded*attention_mask.squeeze().unsqueeze(-1), dim=1)
        elif self.pooling_mode == 'FLATTEN':
            encoded = encoded.reshape((encoded.shape[0], -1)) # B x (L*H)
        elif self.pooling_mode == 'CLS': # CLS token needs to be prepended
            encoded = encoded[:,0,:]
        encoded = self.linear1(encoded)
        encoded = self.relu(encoded)
        encoded = self.linear2(encoded)
        return encoded

class ClassificationModel(nn.Module):
    """ FNSFormer classificaton model.
    """
    def __init__(self, config):
        """
        FNSEncoder + classifier head.
        """
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.padding_idx = config["padding_idx"]
        # ----- add models here -----
        if config['model_name'] == 'fnsformer':
            self.encoder = FNSEncoder(config)
        if config['model_name'] == 'opfnsformer':
            self.encoder = OPFNSEncoder(config)            
        elif config['model_name'] == 'dpformer':
            self.encoder = DPEncoder(config)    
        elif config['model_name'] == 'sinkformer':
            self.encoder = SINKEncoder(config)    
        # else:
        #     model_name = config['model_name']
        #     print(f'Model {model_name} does not exist!')
        #     quit()
        self.classifier = ClassifierHead(config)

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Expand attention mask if CLS is prepended
        if self.pooling_mode == 'CLS':
            # B x 1 x 1 x L+1
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1, 1, 1).to(x.device), attention_mask], dim=-1)
        # Find last non-padded index (CHECK)
        B, L = x.shape
        last_nonpadded_idx = (torch.arange(B) * L).to(x.device) + (torch.sum(x != self.padding_idx, dim=1, dtype=torch.int)-1).to(x.device) # B
        # Calculate the encoder's output
        x, all_attentions = self.encoder(
            x, attention_mask=attention_mask, output_attentions=output_attentions
        )
        x = self.classifier(x, attention_mask=attention_mask)
        # If no pooling, take the output for last token?
        if self.classifier.pooling_mode not in ['CLS', 'MEAN', 'SUM', 'FLATTEN']:
            num_classes = x.shape[-1]
            x = x.view(B*L, num_classes)[last_nonpadded_idx].view(B, num_classes) # B x num_classes
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)

class DualClassifierHead(nn.Module):
    """Classifier head for dual encoding problem."""
    # See classifier_head_dual in LRA > lra_benchmarks > models > layers > common_layers.
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.interaction = config["interaction"]
        if self.pooling_mode == 'FLATTEN':
            pool_dim = config["hidden_size"] * config["max_length"]
        else:
            pool_dim = config["hidden_size"]
        if self.interaction == 'NLI':
            in_dim = pool_dim * 4
        else:
            in_dim = pool_dim * 2
        self.linear1 = nn.Linear(in_features=in_dim, out_features=config["intermediate_size"])
        self.relu = nn.ReLU() # CAN CHANGE THIS
        self.linear2 = nn.Linear(in_features=config["intermediate_size"], out_features=config["intermediate_size"])
        self.linear3 = nn.Linear(in_features=config["intermediate_size"], out_features=config["num_classes"])
        
    def forward(self, encoded, attention_mask=None):
        # Pooling across sequence length; 2*B x L x H -> 2*B x H
        if self.pooling_mode == 'MEAN':
            encoded = torch.mean(encoded*attention_mask.squeeze().unsqueeze(-1), dim=1)
        elif self.pooling_mode == 'SUM':
            encoded = torch.sum(encoded*attention_mask.squeeze().unsqueeze(-1), dim=1)
        elif self.pooling_mode == 'FLATTEN':
            encoded = encoded.reshape((encoded.shape[0], -1)) # 2*B x (L*H)
        elif self.pooling_mode == 'CLS':
            encoded = encoded[:,0,:]
        else:
            raise NotImplementedError('Must pool.')
        encoded1, encoded2 = torch.split(encoded, 2, dim=0)
        if self.interaction == 'NLI':
            # NLI interaction style
            encoded = torch.cat([encoded1, encoded2, encoded1 * encoded2,
                                       encoded1 - encoded2], dim=1)
        else:
            encoded = torch.cat([encoded1, encoded2], dim=1)
        encoded = self.linear1(encoded)
        encoded = self.relu(encoded)
        encoded = self.linear2(encoded)
        encoded = self.relu(encoded)
        encoded = self.linear3(encoded)
        return encoded


class RetrievalModel(nn.Module): # CHECK
    """FNSFormer retrieval model for matching (dual encoding) tasks."""
    def __init__(self, config):
        """
        FNSEncoder + mean pooling + MLP.
        """
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        if self.pooling_mode == 'CLS':
            self.cls_token = nn.Parameter(torch.randn(1,1,config["hidden_size"]))
        # ----- add models here -----
        if config['model_name'] == 'fnsformer':
            self.encoder = FNSEncoder(config)
        if config['model_name'] == 'opfnsformer':
            self.encoder = OPFNSEncoder(config)            
        elif config['model_name'] == 'dpformer':
            self.encoder = DPEncoder(config)    
        elif config['model_name'] == 'sinkformer':
            self.encoder = SINKEncoder(config)    
        else:
            model_name = config['model_name']
            print(f'Model {model_name} does not exist!')
            quit()
        self.classifier = DualClassifierHead(config)

    def forward(self, x, attention_mask=None, output_attentions=False):
        # Expand attention mask if CLS is prepended
        if self.pooling_mode == 'CLS':
            # B x 1 x 1 x L+1
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1, 1, 1), attention_mask], dim=-1)
        # Calculate encoder output for both inputs
        x, all_attentions = self.encoder(
            x, attention_mask=attention_mask, output_attentions=output_attentions
        ) # 2*B x L x H
        x = self.classifier(x, attention_mask=attention_mask) # B x num_classes
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)