import argparse
import json
import math
import os
import torch
from importlib.metadata import version as package_version
import CONFIG
from data import Dataset
from modules.transformer import Transformer
from sacremoses import MosesDetokenizer
from utils.experiment import Experiment
from utils.evaluation import (
    build_sacrebleu_metric,
    generated_ids_to_text,
    greedy_decode_from_encoder,
    score_corpus_bleu
)
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

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
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--d_model', default=256, type=int)
    # ----- DP -----
    parser.add_argument('--model_name', default='dp' + MODEL_SUFFIX, type=str)
    # ----- Sinkformer -----
    parser.add_argument('--n_it', default=3, type=int,
                        help='Positive odd number of Sinkhorn row/column normalizations.')
    # ----- FNS -----
    parser.add_argument('--manifold', default='rd', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--bandwidth', default=1, type=float)  
    parser.add_argument('--a', default=0, type=float)
    parser.add_argument('--is_rescale_dist', type=str2bool, default=True)    
    # ----- training -----
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=0, type=float)
    parser.add_argument('--lr_reduction_factor', default=0.3, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--scheduler', default='reduce_on_plateau', type=str,
                        choices=['reduce_on_plateau', 'warmup_cosine', 'noam', 'none'])
    parser.add_argument('--warmup_steps', default=CONFIG.NUM_WARMUP, type=int)
    parser.add_argument('--total_steps', default=0, type=int,
                        help='Total optimizer steps for warmup_cosine. Inferred if unset.')
    parser.add_argument('--steps_per_epoch', default=0, type=int,
                        help='Optimizer steps per epoch for warmup_cosine. Inferred if unset.')
    parser.add_argument('--noam_factor', default=0.5, type=float)
    parser.add_argument('--bleu_max_new_tokens', default=64, type=int,
                        help='Reference-independent generation limit for validation BLEU.')

    args = parser.parse_args()
    args.scheduler = args.scheduler.lower()
    if args.bleu_max_new_tokens < 1:
        parser.error('--bleu_max_new_tokens must be positive')

    print('[~] Training')
    print(f' ~  Using device: {Transformer.device}')

    ##### SET SEED #####
    torch.manual_seed(args.seed)   
    #seed_everything(args.seed)  # somehow not necessarily needed

    # set up model_name
    model_name = args.model_name.lower()

    # Download and preprocess data
    dataset = Dataset(CONFIG.LANGUAGE_PAIR, batch_size=CONFIG.BATCH_SIZE)
    trg_sos_index = dataset.trg_vocab[dataset.sos_token]
    trg_eos_index = dataset.trg_vocab[dataset.eos_token]
    trg_pad_index = dataset.trg_vocab[dataset.pad_token]
    bleu_metric = build_sacrebleu_metric()
    detokenizer = MosesDetokenizer(lang=dataset.trg_lang)

    # IMMUTABLE except for seed, is_op
    config = {'d_model':       args.d_model,
              'src_vocab_len': len(dataset.src_vocab),
              'trg_vocab_len': len(dataset.trg_vocab),
              'src_pad_index': dataset.src_vocab[dataset.pad_token],
              'trg_pad_index': trg_pad_index,
              'num_heads':     args.num_heads,
              'num_layers':    args.num_layers,
              'dropout_rate':  0.1,
              'seed':          args.seed,
              'is_op':         args.is_op,
              'qkv_bias':      args.qkv_bias
              }
    
    if model_name[-9:] == 'fns' + MODEL_SUFFIX:
        model_name = args.manifold + model_name 
        config['alpha'], config['bandwidth'], config['a'] = args.alpha, args.bandwidth, args.a
        config['is_rescale_dist'] = args.is_rescale_dist
    elif model_name == 'sink' + MODEL_SUFFIX:
        if args.n_it < 1 or args.n_it % 2 == 0:
            parser.error('--n_it must be a positive odd integer')
        if not math.isfinite(args.bandwidth) or args.bandwidth != 1:
            parser.error('--bandwidth must be 1 for the paper-compatible Sinkformer')
        config['n_it'], config['bandwidth'] = args.n_it, 1
    config['model_name'] = model_name
    train_config = {'lr': args.lr, 'min_lr': args.min_lr, 
                    'beta1': CONFIG.BETA1, 'beta2': CONFIG.BETA2, 'eps': CONFIG.EPS,
                    'batch_size': CONFIG.BATCH_SIZE, 'epochs': args.num_epochs,
                    'scheduler': args.scheduler, 'warmup_steps': args.warmup_steps,
                    'total_steps': args.total_steps, 'steps_per_epoch': args.steps_per_epoch,
                    'bleu_method': 'autoregressive_greedy_corpus_sacrebleu',
                    'bleu_split': 'validation', 'bleu_decoding': 'greedy',
                    'sacrebleu_version': package_version('sacrebleu'),
                    'sacremoses_version': package_version('sacremoses'),
                    'bleu_tokenizer': '13a', 'bleu_lowercase': True,
                    'bleu_smoothing': 'exp', 'bleu_effective_order': False,
                    'bleu_num_references': 1,
                    'bleu_detokenizer': 'sacremoses',
                    'bleu_max_new_tokens': args.bleu_max_new_tokens,
                    'bleu_scale': '0-100'
    }

    train_config['scheduler'] = args.scheduler
    if args.scheduler == 'reduce_on_plateau':
        train_config['lr_reduction_factor'] = args.lr_reduction_factor
    elif args.scheduler == 'noam':
        train_config['noam_factor'] = args.noam_factor

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

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        #lr=CONFIG.LEARNING_RATE,
        lr=args.lr,        
        betas=(CONFIG.BETA1, CONFIG.BETA2),
        eps=CONFIG.EPS
    )


    def infer_steps_per_epoch():
        num_batches = 0
        for _ in dataset.train_loader:
            num_batches += 1
        if num_batches == 0:
            raise ValueError('Could not infer steps_per_epoch from an empty train loader')
        return num_batches


    def build_scheduler():
        if args.scheduler == 'none':
            return None, 'none', None, None

        if args.scheduler == 'reduce_on_plateau':
            return (
                ReduceLROnPlateau(optimizer, min_lr=args.min_lr, factor=args.lr_reduction_factor,
                                  patience=args.patience),
                'validation_epoch',
                None,
                None
            )

        if args.lr <= 0:
            raise ValueError('--lr must be positive when using a LambdaLR scheduler')
        if args.warmup_steps <= 0:
            raise ValueError('--warmup_steps must be positive')

        if args.scheduler == 'noam':
            def noam_lr_lambda(step):
                step = max(step + 1, 1)  # LambdaLR step is zero-indexed.
                lr = args.noam_factor * (args.d_model ** -0.5) * min(
                    step ** -0.5,
                    step * (args.warmup_steps ** -1.5)
                )
                return lr / args.lr

            return LambdaLR(optimizer, noam_lr_lambda), 'train_batch', None, None

        steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else infer_steps_per_epoch()
        total_steps = args.total_steps if args.total_steps > 0 else args.num_epochs * steps_per_epoch
        if total_steps <= args.warmup_steps:
            raise ValueError('--total_steps must be greater than --warmup_steps for warmup_cosine')

        min_lr_ratio = args.min_lr / args.lr
        if min_lr_ratio > 1:
            raise ValueError('--min_lr must be less than or equal to --lr for warmup_cosine')

        def warmup_cosine_lr_lambda(step):
            step = max(step + 1, 1)  # LambdaLR step is zero-indexed.
            if step <= args.warmup_steps:
                return max(step / args.warmup_steps, min_lr_ratio)

            progress = min((step - args.warmup_steps) / (total_steps - args.warmup_steps), 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, warmup_cosine_lr_lambda), 'train_batch', steps_per_epoch, total_steps


    scheduler, scheduler_step, resolved_steps_per_epoch, resolved_total_steps = build_scheduler()
    train_config['scheduler_step'] = scheduler_step
    if resolved_steps_per_epoch is not None:
        train_config['steps_per_epoch'] = resolved_steps_per_epoch
    if resolved_total_steps is not None:
        train_config['total_steps'] = resolved_total_steps
    print(f' ~  Scheduler: {args.scheduler} ({scheduler_step})')

    # Save config, train_config
    with open(njoin(out_dir,"config.json"), "w") as ofile:
        json.dump(config, ofile)
    with open(njoin(out_dir,"train_config.json"), "w") as ofile:
        json.dump(train_config, ofile)

    # Cross entropy loss
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=config['trg_pad_index'])


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
            if scheduler_step == 'train_batch':
                scheduler.step()

            train_loss += loss.item()
            num_batches += 1
            del src, trg

        experiment.add_scalar('loss/train', epoch, train_loss / num_batches)
        validate(epoch)


    # Evaluate validation loss with teacher forcing, but generate translations
    # autoregressively for one corpus-level SacreBLEU score.
    def validate(epoch):
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            num_batches = 0
            hypotheses = []
            references = []
            for data in dataset.valid_loader:
                src = data['source'].to(model.device)
                trg = data['target'].to(model.device)

                enc_out, source_padding_mask = model.encode(src)
                predictions = model.decode(
                    trg[:, :-1],
                    enc_out,
                    source_padding_mask
                )

                loss = loss_function(
                    predictions.reshape(-1, predictions.size(-1)),
                    trg[:, 1:].reshape(-1)
                )

                generated = greedy_decode_from_encoder(
                    model,
                    enc_out,
                    source_padding_mask,
                    trg_sos_index,
                    trg_eos_index,
                    trg_pad_index,
                    args.bleu_max_new_tokens
                ).cpu()
                batch_references = data['target_text']
                if len(batch_references) != generated.size(0):
                    raise ValueError('target_text batch does not match generated translations')

                for token_ids, reference in zip(generated, batch_references):
                    hypotheses.append(
                        generated_ids_to_text(
                            token_ids,
                            dataset.trg_vocab,
                            detokenizer,
                            trg_sos_index,
                            trg_eos_index,
                            trg_pad_index
                        )
                    )
                    references.append(reference.strip())

                valid_loss += loss.item()
                num_batches += 1
                del src, trg, enc_out, predictions, generated

            if num_batches == 0:
                raise ValueError('Cannot validate with an empty validation loader')
            valid_loss /= num_batches
            bleu_result = score_corpus_bleu(
                bleu_metric,
                hypotheses,
                references
            )
            if scheduler_step == 'validation_epoch':
                scheduler.step(valid_loss)

            experiment.add_scalar('loss/validation', epoch, valid_loss)
            experiment.add_scalar('bleu', epoch, bleu_result.score)
            experiment.add_scalar('lr', epoch, next(iter(optimizer.param_groups))['lr'])
            if epoch == 0:
                print(f' ~  SacreBLEU signature: {bleu_metric.get_signature()}')

    # quit()  # delete
    experiment.loop(args.num_epochs, train)
