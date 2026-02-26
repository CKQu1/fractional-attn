import argparse
import json
import os
import torch
import CONFIG
from data import Dataset
from modules import Transformer
from nltk.translate.bleu_score import sentence_bleu
#import sacrebleu
from utils.experiment import Experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import MODEL_SUFFIX, DROOT
from utils.mutils import str2bool, njoin, structural_model_root, create_model_dir

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main.py training arguments')  
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_root', default=njoin(DROOT, 'experiments'), type=str, help='root dir of storing the model')
    # ----- Model general -----
    parser.add_argument('--is_op', type=str2bool, default=False)
    parser.add_argument('--qkv_bias', type=str2bool, default=False)
    # ----- DP -----
    parser.add_argument('--model_name', default='dp' + MODEL_SUFFIX, type=str)
    # ----- FNS -----
    parser.add_argument('--manifold', default='rd', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)  
    parser.add_argument('--a', default=0, type=float)
    parser.add_argument('--is_rescale_dist', type=str2bool, default=True)    
    # ----- training -----
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=0, type=float)
    parser.add_argument('--lr_reduction_factor', default=0.3, type=float)

    args = parser.parse_args()

    print('[~] Training')
    print(f' ~  Using device: {Transformer.device}')

    ##### SET SEED #####
    torch.manual_seed(args.seed)   
    #seed_everything(args.seed)  # somehow not necessarily needed

    # set up model_name
    model_name = args.model_name.lower()

    # Download and preprocess data
    dataset = Dataset(CONFIG.LANGUAGE_PAIR, batch_size=CONFIG.BATCH_SIZE)

    # IMMUTABLE except for seed, is_op
    config = {'d_model':       CONFIG.D_MODEL,
              'src_vocab_len': len(dataset.src_vocab),
              'trg_vocab_len': len(dataset.trg_vocab),
              'src_pad_index': dataset.src_vocab[dataset.pad_token],
              'trg_pad_index': dataset.trg_vocab[dataset.pad_token],
              'num_heads':     8,
              'num_layers':    6,
              'dropout_rate':  0.1,
              'seed':          args.seed,
              'is_op':         args.is_op,
              'qkv_bias':      args.qkv_bias
              }
    
    if model_name[-9:] == 'fns' + MODEL_SUFFIX:
        model_name = args.manifold + model_name 
        config['alpha'], config['bandwidth'], config['a'] = args.alpha, args.bandwidth, args.a
        config['is_rescale_dist'] = args.is_rescale_dist
    config['model_name'] = model_name
    train_config = {'lr': args.lr, 'min_lr': args.min_lr, 
                    'lr_reduction_factor': args.lr_reduction_factor,
                    'beta1': CONFIG.BETA1, 'beta2': CONFIG.BETA1, 'eps': CONFIG.EPS,
                    'batch_size': CONFIG.BATCH_SIZE, 'epochs': CONFIG.NUM_EPOCHS 
    }

    # Initialize model
    model = Transformer(config)

    print(f' ~  Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Set up saving path
    if args.model_root == '':
        model_root = structural_model_root(qk_share=args.qk_share, n_layers=args.n_layers,
                                           n_attn_heads=args.n_attn_heads, hidden_size=args.hidden_size  # lr=args.lr, bs=args.train_bs,                                                                                          
                                           )       
        model_root = njoin(DROOT, model_root)
    else:
        model_root = args.model_root      
    models_dir, out_dir = create_model_dir(model_root, **config, category='-'.join(CONFIG.LANGUAGE_PAIR))      

    # Set up experiment
    experiment = Experiment(model, root=out_dir)

    # Save config, train_config
    with open(njoin(out_dir,"config.json"), "w") as ofile: 
        json.dump(config, ofile)    
    with open(njoin(out_dir,"train_config.json"), "w") as ofile: 
        json.dump(train_config, ofile)    

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        #lr=CONFIG.LEARNING_RATE,
        lr=args.lr,        
        betas=(CONFIG.BETA1, CONFIG.BETA2),
        eps=CONFIG.EPS
    )


    #################### OPTION 1 ####################
    # Lambda LR Scheduler as described in paper:

    # def get_lr(x):
    #     x += 1      # x is originally zero-indexed
    #     return (CONFIG.D_MODEL ** (-0.5)) * min(x ** (-0.5), x * (CONFIG.NUM_WARMUP ** (-1.5)))

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # plot LR scheduler
    # import seaborn as sns
    # from matplotlib import pyplot as plt    
    # ax = sns.lineplot(
    #     x=range(CONFIG.NUM_EPOCHS),
    #     y=[get_lr(x) for x in range(CONFIG.NUM_EPOCHS)]
    # )
    # ax.set(xlabel='Epoch', ylabel='Learning Rate', title='Learning Rate Schedule')
    # plt.savefig(os.path.join(experiment.path, 'lr_schedule.png'))    

    #################### OPTION 2 ####################
    # Instead, reducing LR by factor on loss plateau works much, much better
    scheduler = ReduceLROnPlateau(optimizer, min_lr=args.min_lr, factor=args.lr_reduction_factor)

    # Cross entropy loss
    loss_function = torch.nn.CrossEntropyLoss()


    # Train
    def train(epoch):
        model.train()
        train_loss = 0
        num_batches = 0     # Using DataPipe, cannot use len() to get number of batches
        for data in dataset.train_loader:
            src = data['source'].to(model.device)
            trg = data['target'].to(model.device)

            # Given the sequence length N, transformer tries to predict the N+1th token.
            # Thus, transformer must take in trg[:-1] as input and predict trg[1:] as output.
            optimizer.zero_grad()
            predictions = model(src, trg[:, :-1])

            # For CrossEntropyLoss, need to reshape input from (batch, seq_len, vocab_len)
            # to (batch * seq_len, vocab_len). Also need to reshape ground truth from
            # (batch, seq_len) to just (batch * seq_len)
            loss = loss_function(
                predictions.reshape(-1, predictions.size(-1)),
                trg[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            del src, trg

        experiment.add_scalar('loss/train', epoch, train_loss / num_batches)
        validate(epoch)


    # Evaluate against validation set and calculate BLEU
    def validate(epoch):
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            num_batches = 0
            bleu_score = 0
            for data in dataset.valid_loader:
                src = data['source'].to(model.device)
                trg = data['target'].to(model.device)

                predictions = model(src, trg[:, :-1])

                loss = loss_function(
                    predictions.reshape(-1, predictions.size(-1)),
                    trg[:, 1:].reshape(-1)
                )

                # Calculate BLEU score
                batch_size = predictions.size(0)
                batch_bleu = 0
                p_indices = torch.argmax(predictions, dim=-1)
                for i in range(batch_size):
                    p_tokens = dataset.trg_vocab.lookup_tokens(p_indices[i].tolist())
                    t_tokens = dataset.trg_vocab.lookup_tokens(trg[i, 1:].tolist())

                    # Filter out special tokens
                    p_tokens = list(filter(lambda x: '<' not in x, p_tokens))
                    t_tokens = list(filter(lambda x: '<' not in x, t_tokens))

                    if len(p_tokens) > 0 and len(t_tokens) > 0:
                        batch_bleu += sentence_bleu([t_tokens], p_tokens)
                        #batch_bleu += sacrebleu.corpus_bleu([t_tokens], p_tokens).score
                bleu_score += batch_bleu / batch_size

                valid_loss += loss.item()
                scheduler.step(loss.item())
                num_batches += 1
                del src, trg

            experiment.add_scalar('loss/validation', epoch, valid_loss / num_batches)
            experiment.add_scalar('bleu', epoch, bleu_score / num_batches)
            experiment.add_scalar('lr', epoch, next(iter(optimizer.param_groups))['lr'])

    # quit()  # delete
    experiment.loop(CONFIG.NUM_EPOCHS, train)
