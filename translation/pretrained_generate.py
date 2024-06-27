import argparse
import json
import os
import pandas as pd
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from contextlib import nullcontext
from os.path import isdir, isfile
from os import makedirs
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#from model import GPTConfig, GPT

from constants import *
from mutils import njoin, create_model_dir, convert_train_history, structural_model_root
from data_utils import prepare_data, BilingualDataset

from torch.optim import AdamW

from transformers import AutoTokenizer

# import torchmetrics

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

import torchmetrics
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

"""
python -i model_generate.py\
 --out_dir=.droot/test-run-v1/en_layers\=2-de_layers\=2-heads\=None-hidden\=128-qkv/dpnmt-iwslt14-qkv/model\=0/ --split=train --eval_iters=5
"""

if __name__ == '__main__':

    # Training options
    parser = argparse.ArgumentParser(description='translation/ddp_main.py training arguments')   
    
    parser.add_argument('--out_dir', default='', type=str, help='model dir')
    parser.add_argument('--eval_iters', default=5, type=int, help='number of sentences to translate')
    parser.add_argument('--split', default='val', type=str)    

    # Dataset settings
    parser.add_argument('--dataset_name', default='iwslt14', type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)  # tokenizer file path

    args = parser.parse_args()    

    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = args.out_dir      
    train_setting = pd.read_csv(njoin(out_dir, 'train_setting.csv'))
    if os.path.isdir(njoin(out_dir, 'run_performance.csv')):
        run_performance = pd.read_csv(njoin(out_dir, 'run_performance.csv'))
    f = open(njoin(out_dir,'config.json'))
    config = json.load(f)
    f.close()
    f = open(njoin(out_dir,'attn_setup.json'))
    attn_setup = json.load(f)
    f.close()    

    # data
    dataset = attn_setup['dataset_name']
    #train_batch_size = int(train_setting.loc[0,'batch_size'])
    train_batch_size = 1
    eval_batch_size = 1
    # model
    model_name = attn_setup['model_name']
    
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    # DDP settings
    backend = 'nccl' if device == 'cuda' else 'gloo' 

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?    
    if ddp and args.train_with_ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        #init_process_group(backend=backend)
        init_process_group(backend=backend, world_size=ddp_world_size, rank=ddp_rank)  # rank=ddp_local_rank        
        if device == 'cuda':
            #device = f'cuda:{ddp_local_rank}'
            device = f'cuda:{ddp_rank}'
            torch.cuda.set_device(device)
        else:
            device = f'cpu'   
            #torch.cpu.set_device(device)     
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size  
        
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1                 

    torch.manual_seed(1337 + seed_offset)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # else:
    #     torch.backends.cpu.matmul.allow_tf32 = True # allow tf32 on matmul
    #     torch.backends.cpu.allow_tf32 = True # allow tf32 on cudnn        
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def get_all_sentences(ds, lang):
        for item in ds:
            yield item['translation'][lang]
            
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

    if 'iwslt' in dataset: 
        year_dataset = '20' + dataset[dataset.find('iwslt') + 5:]
        src_language, trg_language = 'de', 'en'
        dataset = load_dataset("ted_talks_iwslt", language_pair=(src_language, trg_language), year=year_dataset, split="train")
        tokenizer_src = get_or_build_tokenizer(config, dataset, src_language)
        tokenizer_trg = get_or_build_tokenizer(config, dataset, trg_language)
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.2, shuffle=True) #20% test set
        #trainset, testset = dataset
        trainset = dataset['train']; testset = dataset['test']
        trainset = BilingualDataset(trainset, tokenizer_src, tokenizer_trg, src_language, trg_language, config["max_length"])
        testset = BilingualDataset(testset, tokenizer_src, tokenizer_trg, src_language, trg_language, config["max_length"])
        # Create dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=ddp_world_size)
        testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=ddp_world_size)
        # Get pad token ids and vocab sizes
        config["src_vocab_size"] = tokenizer_src.get_vocab_size()
        config["src_pad_token_id"] = tokenizer_src.token_to_id("[PAD]")
        config["trg_vocab_size"] = tokenizer_trg.get_vocab_size()
        config["trg_pad_token_id"] = tokenizer_trg.token_to_id("[PAD]")

        src_vocab_size = config["src_vocab_size"]
        trg_vocab_size = config["trg_vocab_size"]


    checkpoint = torch.load(njoin(out_dir, 'ckpt.pt'), map_location=device)
    if model_name == 'dpnmt':
        from models.translation import DPForNMT
        model = DPForNMT(config)
    elif model_name == 'sinknmt':
        from models.sink_translation import SINKForNMT
        model = SINKForNMT(config)            
    elif model_name == 'fnsnmt':
        from models.fns_translation import FNSForNMT
        model = FNSForNMT(config)    
    elif model_name == 'opfnsnmt':
        from models.opfns_translation import OPFNSForNMT
        model = OPFNSForNMT(config)       
    elif model_name == 'rdfnsnmt':
        from models.rdfns_translation import RDFNSForNMT
        model = RDFNSForNMT(config)    
    elif model_name == 'rdopfnsnmt':
        from models.rdopfns_translation import RDOPFNSForNMT               
    else:
        print(f'{model_name} does not exist!')
        quit()    

    model.load_state_dict(checkpoint['model'])

    print(f'---------- Model loaded! ----------')

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
                # truncate the src_mask to create the cross-attn mask here
                out, _, _ = model.module.decoder(decoder_input, encoder_output, src_mask=source_mask[:,:,:trg_len,:], src_pad_mask=source_pad_mask,
                                                 trg_mask=trg_mask, trg_pad_mask=trg_pad_mask)
            else:
                # truncate the src_mask to create the cross-attn mask here
                out, _, _ = model.decoder(decoder_input, encoder_output, src_mask=source_mask[:,:,:trg_len,:], src_pad_mask=source_pad_mask,
                                          trg_mask=trg_mask, trg_pad_mask=trg_pad_mask)
            
            #out[0,-1,config['trg_pad_token_id']] = float('-Inf')            
            #out[0,-1,sos_idx] = float('-Inf')

            # get next token
            next_word = out[0,-1,:].argmax(-1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )
            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)        

    special_tokens_dict = {}
    for word in list(tokenizer_trg.get_vocab().keys()):
        if '[' in word and ']' in word:
            special_tokens_dict[word] = tokenizer_trg.get_vocab()[word]

    eval_iters = args.eval_iters
    split = args.split
    model_outs = []
    predicted = []
    expected = []  # greedy decoded
    decoded_texts = []
    losses = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=config["trg_pad_token_id"])

    def get_bleu_score(ref_sentence, pred_sentence):
        hyp = [pred_sentence.split(' ')]
        ref = [ref_sentence.split(' ')]               
        bleu_score = corpus_bleu(ref, hyp, smoothing_function=SmoothingFunction().method4)     
        return bleu_score

    model.eval()    
    for _ in tqdm(range(eval_iters)):
        X, Y, src_mask, src_pad_mask, trg_mask, trg_pad_mask, tgt_text = get_batch(split)
        logits, _, _, _ = model(X, Y, src_mask, src_pad_mask, trg_mask, trg_pad_mask)        
        loss = loss_fn(logits.view(-1, trg_vocab_size), Y.view(-1))
        losses.append(loss)
        bs = X.shape[0]
        for batch_idx in range(bs):
            model_out = greedy_decode(model, X[batch_idx,None], src_mask[batch_idx,None], src_pad_mask[batch_idx,None], 
                                      tokenizer_trg, config["max_length"], device) 
            model_outs.append(model_out)

            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())
            predicted.append(model_out_text)            

            decoded_text = tokenizer_trg.decode(logits.argmax(-1).detach().cpu().numpy()[batch_idx])   
            decoded_texts.append(decoded_text)

            expected.append(tgt_text[batch_idx])

    print(model_outs)
    print('\n')
    print('-'*15 + ' Translation results ' + '-'*15 + '\n')
    for ii in range(len(predicted)):  
        print('<'*40)      
        print(f'Case {ii+1} \n')
        print(f'SENTENCE --> {expected[ii]} \n')
        print(f'DECODED --> {decoded_texts[ii]} \n')  
        print(f'Bleu score: {get_bleu_score(expected[ii], decoded_texts[ii])} \n ')
        print(f'GREEDY DECODED --> {predicted[ii]} \n')    
        print(f'Bleu score: {get_bleu_score(expected[ii], predicted[ii])} \n ')    
        print('>'*40 + '\n')