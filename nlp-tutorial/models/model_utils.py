import numpy as np
import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

def load_glove_embeddings(file_path, tokenizer, embedding_dim):
    from tqdm import tqdm
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.random.normal(size=(tokenizer.vocab_size, embedding_dim))
    
    for word, idx in tqdm(tokenizer.get_vocab().items()):
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        elif word.lower() in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word.lower()]
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)

def load_pretrained_model(config):
    if config['pretrained_model_name'] == 'distilbert-base-uncased':

        pretrained_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    elif config['pretrained_model_name'] == 'glove':

        # Initializing a DistilBERT configuration
        distilbertconfig = DistilBertConfig(n_heads=config['n_heads'], n_layers=config['n_layers'],
                                            dim=config['d_model'], hidden_dim=config['d_ff'],
                                            vocab_size=config['vocab_size'],
                                            max_position_embeddings=config['seq_len'],
                                            sinusoidal_pos_embds=config['sinusoidal_pos_embds'])
        pretrained_model = DistilBertModel(distilbertconfig)  # random weights 

    return pretrained_model

def load_embeddings(config):

    pretrained_model = load_pretrained_model(config)        

    if config['pretrained_model_name'] == 'glove':     
        from constants import DROOT
        from mutils import njoin

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(f'distilbert/distilbert-base-uncased')

        glove_path = njoin(DROOT, 'GLOVE', 'glove.6B.300d.txt')
        embedding_dim = 300
        glove_embeddings = load_glove_embeddings(glove_path, tokenizer, embedding_dim)        
    
        pretrained_model.embeddings.word_embeddings.weight = nn.Parameter(glove_embeddings)

    return pretrained_model.embeddings

def create_longformer_mask(seq_len, window_size, global_token_indices):

    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(seq_len, i + window_size + 1)
        mask[i, left:right] = 1  # Allow attention to neighbors

    for idx in global_token_indices:
        mask[idx, :] = 1  # Global tokens attend everywhere
        mask[:, idx] = 1  # Everyone attends to global tokens    
    
    return mask