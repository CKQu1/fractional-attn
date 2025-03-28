from collections import OrderedDict

vocab = OrderedDict()
ids_to_tokens = OrderedDict()

vocab_file = 'wiki.vocab'
with open(vocab_file, 'r', encoding='utf-8') as reader:
    for i, line in enumerate(reader.readlines()):
        token =line.split()[0]
        vocab[token] = i
for token, id in vocab.items():
    ids_to_tokens[id] = token