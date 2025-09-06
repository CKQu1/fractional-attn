from prenlp.tokenizer import NLTKMosesTokenizer
from tokenization import Tokenizer, PretrainedTokenizer
from torch.utils.data import DataLoader

from UTILS.data_utils import create_examples

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

def load_dataset_and_tokenizer(args, batch_size):

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

        train_size = len(train_loader.dataset)
        eval_size = len(test_loader.dataset)                      
        steps_per_epoch = len(train_loader)   
        num_classes = 2     
    else:
        assert not args.fix_embed, f'fix_embed cannot be done for dataset {args.dataset_name}'
        from transformers import AutoTokenizer
        from UTILS.data_utils import get_datasets, datasets_create_examples

        # Load a tokenizer (Example: BERT)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenized_train_dataset, tokenized_test_dataset = get_datasets(args, tokenizer)        
        train_loader, test_loader = datasets_create_examples(args, 
                                                             tokenized_train_dataset, 
                                                             tokenized_test_dataset)        
        train_size = len(tokenized_train_dataset)
        eval_size = len(tokenized_test_dataset)  
        steps_per_epoch = len(train_loader)
        num_classes = len(tokenized_train_dataset['label'].unique())


    return tokenizer, train_loader, test_loader, train_size, eval_size, steps_per_epoch, num_classes