import torch
from prenlp.tokenizer import NLTKMosesTokenizer
from tokenization import Tokenizer, PretrainedTokenizer
from torch.utils.data import ConcatDataset, DataLoader, random_split

from UTILS.data_utils import create_examples

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

def split_train_test_datasets(train_dataset, test_dataset, trainset_pct, seed=0):
    if trainset_pct is None:
        return train_dataset, test_dataset

    trainset_pct = float(trainset_pct)
    if not 0 < trainset_pct < 1:
        raise ValueError('trainset_pct must be a float between 0 and 1.')

    train_test_dataset = ConcatDataset([train_dataset, test_dataset])
    total_size = len(train_test_dataset)
    train_size = int(total_size * trainset_pct)
    test_size = total_size - train_size
    if train_size == 0 or test_size == 0:
        raise ValueError(
            f'trainset_pct={trainset_pct} creates train_size={train_size} '
            f'and test_size={test_size} from {total_size} examples.'
        )

    generator = torch.Generator().manual_seed(seed)
    return random_split(train_test_dataset, [train_size, test_size], generator=generator)


def load_dataset_and_tokenizer(args, batch_size, trainset_pct=None):

    if trainset_pct is None:
        trainset_pct = getattr(args, 'trainset_pct', None)

    if args.dataset_name == 'imdb':
        if not args.fix_embed:
            # Load tokenizer
            if args.tokenizer_name == 'sentencepiece':
                tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
            else:
                tokenizer = TOKENIZER_CLASSES[args.tokenizer_name]()
                tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file=args.vocab_file)      

        else:
            if args.pretrained_model_name == 'distilbert-base-uncased':
                from transformers import AutoTokenizer, DistilBertModel
                #tokenizer = AutoTokenizer.from_pretrained(f'distilbert/{args.pretrained_model_name}')
                tokenizer = AutoTokenizer.from_pretrained(f'distilbert/distilbert-base-uncased')
                pretrained_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
                vocab_size, pretrained_model_hidden =\
                     pretrained_model.embeddings.word_embeddings.weight.shape
                pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

                assert args.max_len == pretrained_seq_len - 1, f'args.max_len does not match {pretrained_seq_len}!'

            elif args.pretrained_model_name == 'albert-base-v2':
                from transformers import AlbertTokenizer, AlbertModel
                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                pretrained_model = AlbertModel.from_pretrained("albert-base-v2")                
                vocab_size, pretrained_model_hidden =\
                     pretrained_model.embeddings.word_embeddings.weight.shape
                pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

                assert args.max_len == pretrained_seq_len - 1, f'args.max_len does not match {pretrained_seq_len}!'

            elif args.pretrained_model_name == 'gpt2':
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2")                                
                vocab_size, pretrained_model_hidden =\
                     pretrained_model.transformer.wte.weight.shape
                pretrained_seq_len, _ = pretrained_model.transformer.wpe.weight.shape

                assert args.max_len == pretrained_seq_len - 1, f'args.max_len does not match {pretrained_seq_len}!'

            elif args.pretrained_model_name == 'glove':

                from torchtext.data.utils import get_tokenizer
                from torchtext.vocab import GloVe
                from constants import GLOVE_DIMS
                
                if args.hidden in GLOVE_DIMS:
                    print(f'Pretrained dimension {args.hidden} deployed! \n')

                    # tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, 
                    #                                 args.vocab_file=njoin(DROOT, 'GLOVE', f'vocab_npa_d={args.hidden}.npy'))
                    tokenizer = get_tokenizer("basic_english")
                    glove = GloVe(name='6B', dim=args.hidden)
                    glove_dim = args.hidden
                else:
                    assert args.hidden < max(GLOVE_DIMS), 'Maximal glove dim is 300!'
                    print(f'Reduced dimension {args.hidden} deployed! \n')

                    # for glove_dim in GLOVE_DIMS:
                    #     if glove_dim >= args.hidden:
                    #         break
                    glove_dim = 300
                    tokenizer = get_tokenizer("basic_english")                    
                    glove = GloVe(name='6B', dim=glove_dim)
                vocab_size = len(glove.stoi)    

            #max_length = tokenizer.model_max_length - 1
            #max_length = tokenizer.model_max_length - 2
            #args.max_len = tokenizer.model_max_length - 2
            #args.max_len = tokenizer.model_max_length - 1

            # def preprocess_function(examples):
            #     return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

        # Build DataLoader
        if args.fix_embed and args.pretrained_model_name == 'glove':
            from UTILS.data_utils import glove_create_examples
            train_dataset = glove_create_examples(args, glove_dim, tokenizer, mode='train')
            test_dataset = glove_create_examples(args, glove_dim, tokenizer, mode='test')            
        else:
            train_dataset = create_examples(args, tokenizer, mode='train')
            test_dataset = create_examples(args, tokenizer, mode='test')

        if len(train_dataset) == 0:
            raise ValueError(
                "Training split is empty after loading IMDb data. "
                "Check the local '.data/aclImdb' contents or the fallback archive."
            )
        if len(test_dataset) == 0:
            raise ValueError(
                "Test split is empty after loading IMDb data. "
                "Check the local '.data/aclImdb' contents or the fallback archive."
            )

        train_dataset, test_dataset = split_train_test_datasets(
            train_dataset,
            test_dataset,
            trainset_pct,
            seed=getattr(args, 'seed', 0),
        )
        print('train dataset of size %d' % len(train_dataset))
        print('test dataset of size %d' % len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

        train_size = len(train_loader.dataset)
        eval_size = len(test_loader.dataset)                      
        steps_per_epoch = len(train_loader)   
        num_classes = 2     
    else:
        assert not args.fix_embed, f'fix_embed cannot be done for dataset {args.dataset_name}'
        from UTILS.data_utils import (
            get_datasets,
            datasets_create_examples,
            get_hf_dataset_config,
            set_huggingface_cache_dir,
            as_bool,
        )

        dataset_config = get_hf_dataset_config(args.dataset_name)
        tokenizer_name_arg = getattr(args, 'tokenizer_name', None)
        if tokenizer_name_arg == 'sentencepiece':
            tokenizer_name_arg = None
        hf_tokenizer_name = (
            getattr(args, 'hf_tokenizer_name', None)
            or tokenizer_name_arg
            or dataset_config['tokenizer_name']
        )
        tokenizer_cache_dir = getattr(args, 'tokenizer_cache_dir', '.droot/hf_cache_dir')
        local_files_only = as_bool(getattr(args, 'local_files_only', False))
        tokenizer_cache_dir = set_huggingface_cache_dir(
            tokenizer_cache_dir,
            local_files_only=local_files_only,
        )

        from transformers import AutoTokenizer

        # WordPiece from bert-base-uncased is a common AG News/BERT-family baseline tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            hf_tokenizer_name,
            cache_dir=tokenizer_cache_dir,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized_train_dataset, tokenized_test_dataset, num_classes, label_key = get_datasets(args, tokenizer)
        tokenized_train_dataset, tokenized_test_dataset = split_train_test_datasets(
            tokenized_train_dataset,
            tokenized_test_dataset,
            trainset_pct,
            seed=getattr(args, 'seed', 0),
        )
        print('train dataset of size %d' % len(tokenized_train_dataset))
        print('test dataset of size %d' % len(tokenized_test_dataset))
        train_loader, test_loader = datasets_create_examples(args, 
                                                             tokenized_train_dataset, 
                                                             tokenized_test_dataset,
                                                             label_key=label_key)
        train_size = len(tokenized_train_dataset)
        eval_size = len(tokenized_test_dataset)  
        steps_per_epoch = len(train_loader)


    return tokenizer, train_loader, test_loader, train_size, eval_size, steps_per_epoch, num_classes
