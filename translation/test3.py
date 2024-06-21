import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from datasets import load_dataset

from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from data_utils import prepare_data, BilingualDataset


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

device = f'cuda' if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

train_batch_size = eval_batch_size = 3
config = {
    "beta": 1,
    "bandwidth": 1,
    "sphere_radius": 1,
    "hidden_size": 128,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "num_heads": 1,
    "intermediate_size": 128,
    "hidden_dropout_prob": 0,
    "encoder_dropout_prob": 0,
    "decoder_dropout_prob": 0,
    "attention_probs_dropout_prob": 0,
    "initializer_range": 0.1,
    "qkv_bias": False,
    "use_faster_attention": True,
    # "src_vocab_size": tokenizer_src.vocab_size,
    # "src_pad_token_id": tokenizer_src.pad_token_id,
    # "trg_vocab_size": tokenizer_trg.vocab_size,
    # "trg_pad_token_id": tokenizer_trg.pad_token_id,
    "max_length": 128,
    "tokenizer_path": None
}

src_language, trg_language = 'de', 'en'
dataset = load_dataset("ted_talks_iwslt", language_pair=(src_language, trg_language), year="2014", split="train")
def get_or_build_tokenizer(config, ds, lang):
    if config['tokenizer_path'] is None:
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # tokenizer.save(str(tokenizer_path))
    else:
        tokenizer_path = Path(config['tokenizer_path'].format(lang))
        assert Path.exists(tokenizer_path), "Tokenizer path does not exist"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

tokenizer_src = get_or_build_tokenizer(config, dataset, src_language)
tokenizer_trg = get_or_build_tokenizer(config, dataset, trg_language)


# Split dataset
dataset = dataset.train_test_split(test_size=0.2, shuffle=True) #20% test set
#trainset, testset = dataset
trainset = dataset['train']; testset = dataset['test']
trainset = BilingualDataset(trainset, tokenizer_src, tokenizer_trg, src_language, trg_language, config["max_length"])
testset = BilingualDataset(testset, tokenizer_src, tokenizer_trg, src_language, trg_language, config["max_length"])
# Create dataloaders
ddp_world_size = 1
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=ddp_world_size)
testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=ddp_world_size)
# Get pad token ids and vocab sizes
config["src_vocab_size"] = tokenizer_src.get_vocab_size()
config["src_pad_token_id"] = tokenizer_src.token_to_id("[PAD]")
config["trg_vocab_size"] = tokenizer_trg.get_vocab_size()
config["trg_pad_token_id"] = tokenizer_trg.token_to_id("[PAD]")

##### DP-former #####
from models.translation import DPForNMT
model = DPForNMT(config)

##### FNS/OPFNS-former #####
# config['alpha'] = 1.5
# config['bandwidth'] = 1
# config['a'] = 0

# if config['alpha'] < 2:            
#     config['d_intrinsic'] = int(config['hidden_size'] / config['num_heads'])  # head_dim
# #     config['sphere_radius'] = ((math.pi**(1/config['d_intrinsic'])-1)/math.pi)                            
# # else:
# #     config['sphere_radius'] = 1       
# # config['mask_val'] = math.pi * config['sphere_radius']   
# from models.rdfns_translation import RDFNSForNMT
# model = RDFNSForNMT(config)

##### SINK-former #####
# from models.sink_translation import SINKForNMT
# config['n_it'] = 1
# model = SINKForNMT(config)

print('\n Model created! \n')

# helps estimate an arbitrarily accurate loss over either split using many batches
def greedy_decode(model, source, source_mask, source_pad_mask, tokenizer_trg, max_len, device):        
    sos_idx = tokenizer_trg.get_vocab()['[SOS]']
    eos_idx = tokenizer_trg.get_vocab()['[EOS]']

    # Precompute the encoder output and reuse it for every step
    if ddp:
        encoder_output, _ = model.module.encoder(source, source_mask, source_pad_mask)
    else:
        encoder_output, _ = model.encoder(source, source_mask, source_pad_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)            
    while True:
        trg_len = decoder_input.size(1)
        if trg_len == max_len:
            break
            
        # Decoder self-attention mask
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(1, 1, trg_len, trg_len)
        trg_pad_mask = (decoder_input != config['trg_pad_token_id']).unsqueeze(0).unsqueeze(0)
        # trg_mask = trg_mask.type_as(source_mask).to(device)

        # calculate output
        if ddp:
            out, _, _ = model.module.decoder(decoder_input, encoder_output, src_mask=source_mask[:,:,:trg_len,:], src_pad_mask=source_pad_mask,
                                             trg_mask=trg_mask, trg_pad_mask=trg_pad_mask)
        else:
            out, _, _ = model.decoder(decoder_input, encoder_output, src_mask=source_mask[:,:,:trg_len,:], src_pad_mask=source_pad_mask,
                                      trg_mask=trg_mask, trg_pad_mask=trg_pad_mask)
        # get next token
        next_word = out[0,-1,:].argmax(-1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def test_get_batch(split):
    if split == 'train':
        data = trainloader
        batch_size = train_batch_size
    else:
        data = testloader
        batch_size = eval_batch_size
    ix = torch.randint(len(data), (batch_size,))
    # x = torch.tensor([data.dataset["encoder_input"][i] for i in ix]) # (B, N) 
    # y = torch.tensor([data.dataset["decoder_input"][i] for i in ix]) # (B, N) 
    # encoder_mask = torch.tensor([data.dataset["attention_mask"][i] for i in ix]) # (B, N) 
    # encoder_mask = (encoder_mask.unsqueeze(-1)@encoder_mask.unsqueeze(1)).view(batch_size, 1, config["max_length"], config["max_length"]) # (B,1,N,N)
    # decoder_mask = torch.stack([(torch.tensor(data.dataset["attention_mask"][i] != 0)).unsqueeze(0).int() & causal_mask(config["max_length"]) for i in ix]) # (B,1,N,N)

    ##### CHAGNES HERE #####
    # this seems slow
    # try:
    encoder_lens = []
    decoder_lens = []
    for i in range(len(data)):
        data_point = data.dataset.__getitem__(i)
        encoder_lens.append(data_point['encoder_input'].shape[0])
        decoder_lens.append(data_point['decoder_input'].shape[0])

    #     print(f'{split} dataset works')
    # except:
    #     print(f'{split} dataset failed')

    return encoder_lens, decoder_lens
        

def get_batch(split):
    if split == 'train':
        data = trainloader
        batch_size = train_batch_size
    else:
        data = testloader
        batch_size = eval_batch_size
    ix = torch.randint(len(data), (batch_size,))

    # this seems slow
    data_points = [data.dataset.__getitem__(i.item()) for i in ix]        
    x = torch.stack([data_points[i]['encoder_input'] for i in range(batch_size)])
    y = torch.stack([data_points[i]['decoder_input'] for i in range(batch_size)])
    # pre_encoder_mask = torch.stack([data_points[i]['encoder_mask'].squeeze() for i in range(batch_size)])
    # encoder_mask = (pre_encoder_mask.unsqueeze(-1)@pre_encoder_mask.unsqueeze(1)).view(batch_size, 1, config["max_length"], config["max_length"]) # (B,1,N,N)
    # decoder_mask = torch.stack([data_points[i]['decoder_mask'] for i in range(batch_size)]) # (B,1,N,N)
    encoder_mask = torch.stack([data_points[i]['encoder_mask'] for i in range(batch_size)])
    #encoder_mask = None
    encoder_pad_mask = torch.stack([data_points[i]['encoder_pad_mask'] for i in range(batch_size)])  # (B,1,1,N), type 1: non-square mask
    decoder_mask = torch.stack([data_points[i]['decoder_mask'] for i in range(batch_size)]) # (B,1,N,N), type 2: square mask
    decoder_pad_mask = torch.stack([data_points[i]['decoder_pad_mask'] for i in range(batch_size)])  # (B,1,1,N), type 1: non-square mask

    tgt_text = [data_points[i]['tgt_text'] for i in range(batch_size)]  # added for BLEU score        

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        encoder_mask, decoder_mask = encoder_mask.pin_memory().to(device, non_blocking=True), decoder_mask.pin_memory().to(device, non_blocking=True)
        encoder_pad_mask, decoder_pad_mask = encoder_pad_mask.pin_memory().to(device, non_blocking=True), decoder_pad_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, encoder_mask, decoder_mask = x.to(device), y.to(device), encoder_mask.to(device), decoder_mask.to(device)
        encoder_pad_mask, decoder_pad_mask = encoder_pad_mask.to(device), decoder_pad_mask.to(device)
    #return x, y, encoder_mask, decoder_mask    
    return x, y, encoder_mask, encoder_pad_mask, decoder_mask, decoder_pad_mask, tgt_text   

for split in ['train', 'val']:
    encoder_lens, decoder_lens = test_get_batch(split)
    print(f'{split} dataset')
    print(f'Unique encoder lens: {list(set(encoder_lens))}')
    print(f'Unique decoder lens: {list(set(decoder_lens))} \n')

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss(ignore_index=config["src_pad_token_id"])

split = 'val'
X, Y, src_mask, src_pad_mask, trg_mask, trg_pad_mask, tgt_text = get_batch(split)

predicted = []
expected = []
out_loss = {}
out_bleu = {}
trg_vocab_size = config['trg_vocab_size']
with ctx:
    ##### CHANGES HERE #####
    logits, _, _, _ = model(X, Y, src_mask, src_pad_mask, trg_mask, trg_pad_mask)
    #loss = loss_fn(logits.reshape(-1,trg_vocab_size), F.one_hot(Y, num_classes=trg_vocab_size).type_as(logits).reshape(-1,trg_vocab_size))           
    loss = loss_fn(logits.reshape(-1,trg_vocab_size), Y.reshape(-1))
    if split == 'val':
        # Generate sentence
        # Type 1 (use greedy_decode directly)
        #model_out = greedy_decode(model, X, encoder_mask, tokenizer_trg, config["max_length"], device) 

        # Type 2 (greedy_decode expanded)  
        source = X
        source_mask = src_mask
        source_pad_mask = src_pad_mask
        max_len = config["max_length"]

        # sos_idx = tokenizer_trg.bos_token_id
        # eos_idx = tokenizer_trg.eos_token_id
        ##### CHANGES HERE #####
        sos_idx = tokenizer_trg.get_vocab()['[SOS]']
        eos_idx = tokenizer_trg.get_vocab()['[EOS]']

        # Precompute the encoder output and reuse it for every step
        encoder_output, _ = model.encoder(source, source_mask, source_pad_mask)
        ##### begin{CHANGE} ######
        #decoder_input = Y
        batch_size = Y.shape[0]        

        model_outs = []
        for batch_idx in range(batch_size):
            # Initialize the decoder input with the sos token
            decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

            ##### end{CHANGE} ######
            while True:
                #print(f'----- decoder_input len: {decoder_input.size(1)} -----')

                if decoder_input.size(1) == max_len:
                    break

                trg_len = decoder_input.shape[1]

                # Decoder self-attention mask
                trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(1, 1, trg_len, trg_len)
                trg_pad_mask = (decoder_input != config['trg_pad_token_id']).unsqueeze(0).unsqueeze(0)                   

                out, _, _ = model.decoder(decoder_input, encoder_output[batch_idx,None], 
                                          src_mask=source_mask[None,batch_idx,:,:trg_len,:], src_pad_mask=source_pad_mask[batch_idx,None],
                                          trg_mask=trg_mask, trg_pad_mask=trg_pad_mask)

                #print(f'out shape: {out.shape}')
                # get next token
                next_word = out[0,-1,:].argmax(-1)
                # NEEDED TO INCLUDE batch_size HERE
                decoder_input = torch.cat(
                    [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
                )                

                if next_word == eos_idx:
                    break

            model_out = decoder_input.squeeze(0)
            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())
            predicted.append(model_out_text)

            model_outs.append(model_out)

        expected = tgt_text        
    
# full decoder forward pass for one step of greedy decoding
batch_idx = 0
x = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)  # decoder_input
trg_len = x.shape[1]
trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(1, 1, trg_len, trg_len)
trg_pad_mask = (x != config['trg_pad_token_id']).unsqueeze(0).unsqueeze(0)

position_ids = torch.arange(0, x.shape[-1]).to(x.device)
position_embeddings = model.decoder.positional_embedding(position_ids)
token_embeddings = model.decoder.token_embedding(x)
# Dropout 
# x = self.dropout(position_embeddings + token_embeddings)  # remove dropout for evaluation
x = model.decoder.dropout(position_embeddings + token_embeddings)
# Calculate the transformer block's output for each block
all_self_attentions = []
all_cross_attentions = []
for block in model.decoder.blocks:
    x, _, _ = block(x, encoder_output[batch_idx,None], src_mask=source_mask[None,batch_idx,:,:trg_len,:], src_pad_mask=source_pad_mask[batch_idx,None],
                    trg_mask=trg_mask, trg_pad_mask=trg_pad_mask,
                    output_attentions=False)              
# Linear layer
x = model.decoder.fc(x)
# Softmax
x_prob = nn.Softmax(dim=-1)(x)

# one step
losses = torch.zeros(1)
losses[0] = loss.item()
out_loss[split] = losses.mean()

import torchmetrics
if split == 'val':

    # Method 1
    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    metric = torchmetrics.text.BLEUScore()    
    bleu = metric(predicted, [expected]).item()
    out_bleu[split] = bleu

    # Method 2
    # from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    # hyp = []
    # for idx in range(len(predicted)):
    #     if predicted[idx] == '':
    #         hyp.append([''] * max_len)
    #     else:
    #         hyp.append(predicted[idx].split(' '))
    # ref = [[expected[idx].split(' ')] for idx in range(len(expected))]    
    # bleu = corpus_bleu(ref, hyp)
    # bleu = sentence_bleu(ref, hyp)
    out_bleu[split] = bleu
    print("BLEU score:", bleu)


"""
split = 'val'
batch_idx = 0
X, Y, src_mask, src_pad_mask, trg_mask, trg_pad_mask, tgt_text = get_batch(split)
model_out = greedy_decode(model, X[batch_idx,None], src_mask[batch_idx,None], src_pad_mask[batch_idx,None], tokenizer_trg, config["max_length"], device) 
model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())
print(model_out_text)
"""