import numpy as np
import torch
from os import makedirs
from os.path import isdir, isfile

embeddings_type = 'bert'

assert embeddings_type in ['bert', 'glove']

if embeddings_type == 'bert':

    from transformers import BertModel, BertTokenizer

    # Load pretrained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize input text
    input_text = "Hello, how are you?"
    input_tokens = tokenizer(input_text, return_tensors='pt')

    word_embeddings = model.embeddings.word_embeddings

elif embeddings_type == 'glove':
    
    from constants import DROOT
    from mutils import njoin

    def load_glove_embeddings(file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    # Example usage: Load 100-dimensional GloVe embeddings
    #glove_file = 'glove.6B.100d.txt'
    glove_file = 'glove.6B.300d.txt'
    save_dir = njoin(DROOT, 'EMBDS', 'GLOVE')
    if not isdir(save_dir): makedirs(save_dir)
    embeddings = load_glove_embeddings(njoin(save_dir, glove_file))

    # Get the embedding for a word
    word = "hello"
    embedding = embeddings.get(word)

    # Print the embedding
    print(embedding)

