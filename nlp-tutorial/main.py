import argparse
import pandas as pd
from prenlp.tokenizer import NLTKMosesTokenizer
from torch.utils.data import DataLoader
from os import makedirs
from os.path import isdir

from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
from trainer import Trainer

from constants import DROOT
from path_names import njoin, get_instance

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

# def main(args):
#     print(args)

#     # Load tokenizer
#     if args.tokenizer == 'sentencepiece':
#         tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
#     else:
#         tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
#         tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file =args.vocab_file)
#     # Build DataLoader
#     train_dataset = create_examples(args, tokenizer, mode='train')
#     test_dataset = create_examples(args, tokenizer, mode='test')
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
#     # Build Trainer
#     trainer = Trainer(args, train_loader, test_loader, tokenizer)

#     # Train & Validate
#     for epoch in range(1, args.epochs+1):
#         trainer.train(epoch)
#         trainer.validate(epoch)
#         trainer.save(epoch, args.output_model_prefix)


# quick run (single unit)
"""
python -i main_seq_classification.py  --with_frac=True --gamma=0.5 --max_steps=2 --logging_steps=2 --save_steps=2 --eval_steps=2\
 --divider=100 --warmup_steps=0 --gradient_accumulation_steps=1 --dataset_name=rotten_tomatoes\
 --models_dir=droot/debug_all/debug_mode18/model_l2_0
"""

# quick torchrun (multi-unit)
"""
singularity exec --home ${PBS_O_WORKDIR} ${cpath} torchrun --nproc_per_node=2\
 main.py --dataset=imdb --epochs=10 --n_attn_heads=1 --batch_size=2 --max_seq_len=64 --divider=1000\
 --gradient_accumulation_steps=4\
 --per_device_train_batch_size=4 --per_device_eval_batch_size=4
"""

# under singularity shell
"""
torchrun --nproc_per_node=2\
 main.py --dataset=imdb --epochs=10 --n_attn_heads=1 --batch_size=2 --max_seq_len=64 --divider=1000\
 --gradient_accumulation_steps=4\
 --per_device_train_batch_size=4 --per_device_eval_batch_size=4
"""


# !python -i main.py --dataset=imdb --epochs=10 --n_attn_heads=1 --batch_size=2 --max_seq_len=64 --divider=1000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',             default='imdb',           type=str, help='dataset')
    parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--tokenizer',           default='sentencepiece',  type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER_CLASSES.keys()))
    parser.add_argument('--pretrained_model',    default='wiki.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--output_model_prefix', default='model',          type=str, help='output model name prefix')
    parser.add_argument('--models_dir',          default='',              type=str, help='root dir of storing the model')
    # Input parameters
    parser.add_argument('--batch_size',     default=32,   type=int,   help='batch size')
    parser.add_argument('--divider',     default=1,   type=int,   help='divide the dataset by this number')
    parser.add_argument('--max_seq_len',    default=512,  type=int,   help='the maximum size of the input sequence')
    # Train parameters
    parser.add_argument('--epochs',         default=7,   type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda',        action='store_true')
    # Self-attention parameters
    parser.add_argument('--beta',           default=1,  type=float,   help='fractional power')
    parser.add_argument('--bandwidth',      default=1,  type=float,   help='bandwidth of the kernel')
    # Model parameters
    parser.add_argument('--hidden',         default=256,  type=int,   help='the number of expected features in the transformer')
    parser.add_argument('--n_layers',       default=1,    type=int,   help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_attn_heads',   default=4,    type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--dropout',        default=0.1,  type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden',     default=256, type=int,   help='the dimension of the feedforward network')
    
    args = parser.parse_args()
    
    # main(args)

    print(args)

    # Load tokenizer
    print('Load tokenizer \n')
    if args.tokenizer == 'sentencepiece':
        tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
    else:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
        tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file =args.vocab_file)
    # Build DataLoader
    print('Build DataLoader')
    train_dataset = create_examples(args, tokenizer, mode='train')
    test_dataset = create_examples(args, tokenizer, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    print(f'Train size: {len(train_loader)}; Test size: {len(test_loader)} \n')
    
    # Build Trainer
    print('Build Trainer')
    trainer = Trainer(args, train_loader, test_loader, tokenizer)

    # paths
    if args.models_dir == '':
        root = njoin(DROOT, 'fnsformer_v20240318')
    else:
        root = args.models_dir
    model_suffix = f'_beta={args.beta}_eps={args.bandwidth}'
    instance = get_instance(root, model_suffix)            
    model_dir = njoin(root, f'model={instance}' + model_suffix)
    if not isdir(model_dir): makedirs(model_dir)

    # Train & Validate    
    columns = ['train_loss','train_acc','train_seconds','test_loss','test_acc','test_seconds']
    performance_all = [[] for _ in range(len(columns))]
    df = pd.DataFrame(columns=columns)
    for epoch in range(1, args.epochs+1):
        train_losses_b, train_acc_ns, train_secs, _, = trainer.train(epoch)        
        test_losses_b, test_acc_ns, test_secs = trainer.validate(epoch)
        for ii, item in enumerate([train_losses_b, train_acc_ns, train_secs, test_losses_b, test_acc_ns, test_secs]):
            performance_all[ii].append(item)
        trainer.save(epoch, args.output_model_prefix,
                     model_dir)        
        
    # Save model
    for cidx, colname in enumerate(columns):
        df.loc[:,colname] = performance_all[cidx]
    df.to_csv(njoin(model_dir, 'performance.csv'))
    print(df)
    print(f'Model data saved in {model_dir}')

