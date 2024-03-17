# ----- Dijkstra propagation -----

# import torch
# from dijkstra import DijkstraPQ

# # Debug dijkstra.py

# batch_size, n_heads, q_len = 2, 3, 10
# adj = torch.rand([batch_size, n_heads, q_len, q_len])

# dijk = DijkstraPQ()
# distance_matrix = dijk(adj)

# ----- Model propagation -----

# import torch
# from model import TransformerEncoder

# #Test run for forward propagation.

# beta = 1
# bandwidth = 1

# batch_size = 2
# vocab_size = 200; seq_len = 32
# d_model = 512; n_layers = 1
# n_heads = 1
# encoder = TransformerEncoder(beta, bandwidth, 
#                              vocab_size=vocab_size, seq_len=seq_len,
#                              d_model=d_model,n_layers=n_layers,
#                              n_heads=n_heads 
#                              )
# inp = torch.arange(seq_len).repeat(batch_size, 1)
# outp = encoder(inp)

# ----- Dataset process -----

# import argparse
# from data_utils import create_examples
# from tokenization import Tokenizer, PretrainedTokenizer

# parser = argparse.ArgumentParser()

# parser.add_argument('--dataset',             default='imdb',           type=str, help='dataset')
# parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
# parser.add_argument('--tokenizer',           default='sentencepiece',  type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER_CLASSES.keys()))

# args = parser.parse_args()

# # Load tokenizer
# print('Load tokenizer')
# if args.tokenizer == 'sentencepiece':
#     tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
# else:
#     tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
#     tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file =args.vocab_file)
# # Build DataLoader
# print('Build DataLoader')
# train_dataset = create_examples(args, tokenizer, mode='train')
# test_dataset = create_examples(args, tokenizer, mode='test')


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',             default='imdb',           type=str, help='dataset')
parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
#parser.add_argument('--tokenizer',           default='sentencepiece',  type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER_CLASSES.keys()))
parser.add_argument('--tokenizer',           default='sentencepiece',  type=str)

parser.add_argument('--max_seq_len',    default=32,  type=int,   help='the maximum size of the input sequence')

args = parser.parse_args()