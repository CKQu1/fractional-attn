import argparse
import os
from os import makedirs
from os.path import isdir, isfile
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase
import numpy as np
import torch
from torch.nn import functional as F  
from UTILS.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from constants import HYP_CMAP, HYP_CNORM, MODEL_SUFFIX, DROOT
from models.rdfnsformer import RDFNSformer
from models.dpformer import DPformer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from UTILS.data_utils import glove_create_examples
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx
import torch.nn as nn
    
def eval_model(model):
    _loss = 0
    _acc = 0
    corrects = []
    lengths = []
    for batch in test_dataset:
        inputs, labels = batch
        
        inputs = inputs.to(device)
        labels = labels.to(device)            

        if is_dp: # opdpformer
            outputs, _ = model(inputs[None])
        elif is_fns:
            outputs, _, _ = model(inputs[None])
        outputs = outputs[0]
        
        loss = loss_fn(outputs, labels)
        _loss += loss.item()
        prediction = outputs.argmax(dim=-1)
        correct = prediction == labels
        acc = correct.sum()
        _acc += acc.item()   
        corrects.append(correct)    
        lengths.append(config['max_len'] - count_trailing_zeros(inputs))
    _acc = _acc / len(test_loader)
    _loss = _loss / len(test_loader)
    return _acc , _loss, corrects, lengths
    

if __name__ == '__main__':

    # Configs
    parser = argparse.ArgumentParser(description='nlp-tutorial/fdm.py arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    # parser.add_argument('--models_root', default='', help='Pretrained models root')
    # parser.add_argument('--fns_type', default='rdfnsformer')  # 'spopfns'+MODEL_SUFFIX
    parser.add_argument('--model_dir', default='', help='Pretrained models seed dir')
    parser.add_argument('--use_same_token', type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()   

    # ----- set up -----
    device = f'cuda' if torch.cuda.is_available() else "cpu"
    print(f'device: {device} \n')
    loss_fn = nn.CrossEntropyLoss()      

    model_dir = args.model_dir.replace('\\','')

    # if training is incomplete
    checkpoint = njoin(model_dir, 'ckpt.pt')
    if not isfile(checkpoint):
        print(f'Checkpoint does not exist for {model_dir}')
        quit()

    # ----- load pretrained_model -----    
    attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
    is_fns = attn_setup['model_name'][-9:] == 'fns' + MODEL_SUFFIX
    is_dp = attn_setup['model_name'][-9:] == 'dp' + MODEL_SUFFIX

    seed = attn_setup ['seed']
    fix_embed = attn_setup['fix_embed']
    pretrained_model_name = config['pretrained_model_name'] if fix_embed else False

    config['dataset_name'] = attn_setup['dataset_name']
    config['max_len'] = config['seq_len']
    #d = config['head_dim']
    # manifold = attn_setup['manifold']
    # if 'is_rescale_dist' not in config.keys():
    #     # maybe add prompt
    #     config['is_rescale_dist'] = args.is_rescale_dist  # manual setup            

    main_args = AttrDict(config)               

    # load dataset               
    #batch_size = int(train_setting.loc[0,'batch_size'])
    batch_size = 1    
    if fix_embed:            
        if pretrained_model_name == 'glove':
            from constants import GLOVE_DIMS
            for glove_dim in GLOVE_DIMS:
                if glove_dim >= config['hidden']:
                    break                
            tokenizer = get_tokenizer("basic_english")
            glove = GloVe(name='6B', dim=glove_dim)
            vocab_size = len(glove.stoi)   
            train_dataset = glove_create_examples(main_args, glove_dim, tokenizer, mode='train')
            test_dataset = glove_create_examples(main_args, glove_dim, tokenizer, mode='test')  

        elif pretrained_model_name == 'distilbert-base-uncased':
            from transformers import AutoTokenizer, DistilBertModel
            from UTILS.data_utils import create_examples
            #tokenizer = AutoTokenizer.from_pretrained(f'distilbert/{args.pretrained_model_name}')
            tokenizer = AutoTokenizer.from_pretrained(f'distilbert/distilbert-base-uncased')
            pretrained_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            vocab_size, pretrained_model_hidden =\
                pretrained_model.embeddings.word_embeddings.weight.shape
            pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

            train_dataset = create_examples(main_args, tokenizer, mode='train')
            test_dataset = create_examples(main_args, tokenizer, mode='test')

        elif pretrained_model_name == 'albert-base-v2':
            from transformers import AlbertTokenizer, AlbertModel
            from UTILS.data_utils import create_examples
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            pretrained_model = AlbertModel.from_pretrained("albert-base-v2")                
            vocab_size, pretrained_model_hidden =\
                pretrained_model.embeddings.word_embeddings.weight.shape
            pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

            train_dataset = create_examples(main_args, tokenizer, mode='train')
            test_dataset = create_examples(main_args, tokenizer, mode='test')    

    else:
        from tokenization import Tokenizer, PretrainedTokenizer
        from UTILS.data_utils import create_examples
        tokenizer = PretrainedTokenizer(pretrained_model='wiki.model', vocab_file='wiki.vocab')

        train_dataset = create_examples(main_args, tokenizer, mode='train')
        test_dataset = create_examples(main_args, tokenizer, mode='test')                    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

    # --------------------------------------------------------------
    
    is_fns = attn_setup['model_name'][-9:] == 'fns' + MODEL_SUFFIX 
    if not is_fns: # opdpformer
        model = DPformer(config, is_return_dist=True)
    else: # rdfnsformer
        model = RDFNSformer(config, is_return_dist=True) 

    ckpt = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Random downsample
    np.random.seed(0)
    N_sample = 100
    sample_idxs = np.random.choice(len(train_dataset), N_sample, replace=False)
    # Results
    X_lens = np.zeros(len(sample_idxs))
    spectrum = np.zeros((len(sample_idxs), config['max_len']))
    top_left_eigvecs = np.zeros((len(sample_idxs), config['max_len']))
    dist_matrices = np.zeros((len(sample_idxs), config['max_len'], config['max_len']))
    hop_matrices = np.zeros((len(sample_idxs), config['max_len'], config['max_len']))
    all_weights = np.zeros((len(sample_idxs), config['max_len'], config['max_len']))
    res_idx = 0
    for ii in tqdm(range(len(train_dataset))):
        if ii not in sample_idxs:
            continue
        X, Y = train_dataset[ii]
        X_len = config['max_len'] - count_trailing_zeros(X)
        if is_dp: # opdpformer
            _, attention_weights = model(X[None])  
        else:
            _, attention_weights, g_dists = model(X[None])  
        for lidx, g_dist in enumerate(attention_weights):
            # Attention weights
            if args.use_same_token:
                data = attention_weights[lidx][0,0,:X_len,:X_len].detach().numpy()
            else: # Remove attention of token to itself
                data = attention_weights[lidx][0,0,:X_len,:X_len].detach().numpy()
                np.fill_diagonal(data, 0)
            all_weights[res_idx, :X_len, :X_len] = data 
            # Spectrum
            eigvals, eigvecs = np.linalg.eig(data.T)
            spectrum[res_idx, :len(eigvals)] = eigvals
            top_left_eigvecs[res_idx, :eigvecs.shape[0]] = eigvecs[:,-1].T
            # Shortest path
            edge_weights = 1/np.abs(data)
            np.fill_diagonal(edge_weights, 0)
            G = nx.from_numpy_array(edge_weights, create_using=nx.DiGraph)
            dist_matrix = np.full(data.shape, np.inf)
            hop_matrix = np.full(data.shape, np.inf)
            np.fill_diagonal(dist_matrix, 0)
            np.fill_diagonal(hop_matrix, 0)
            for source in range(dist_matrix.shape[0]):
                lengths, paths = nx.single_source_dijkstra(G, source) # Compute shortest paths using Dijkstra's algorithm
                for target in lengths:
                    dist_matrix[source, target] = lengths[target]
                    hop_matrix[source, target] = len(paths[target]) - 1  # number of edges = nodes - 1
            dist_matrices[res_idx, :X_len, :X_len] = dist_matrix
            hop_matrices[res_idx, :X_len, :X_len] = hop_matrix
            X_lens[res_idx] = X_len
            res_idx += 1
    
    # ----- result_retrieval.py part 1 -----
    max_len = config['max_len']
    attn_weights = np.zeros((max_len, max_len))
    shortest_path = np.zeros((max_len, max_len))    
    # Get results for first full-length sequence
    idx = np.where(X_lens == X_lens.max())[0][0]
    attn_weights[:, :] = all_weights[idx, :, :] # Attention weights
    shortest_path[:, :] = hop_matrices[idx, :, :] # Shortest path    
    # locate the particular sequence
    seq_idx = sample_idxs[idx] # Sequence index
    X, Y = train_dataset[seq_idx] # Should be the corresponding sequence    

    # ------ result_retrieval.py part 2 -----
    mean_spectral_gaps = []
    mean_shortest_paths = []    

    spectral_gap = spectrum[:, -1] - spectrum[:, -2]
    mean_spectral_gap = np.mean(spectral_gap)
    mean_spectral_gaps.append(mean_spectral_gap)
    # Find mean shortest path
    total, n_tokens = 0, 0
    for idx, X_len in enumerate(X_lens):
        X_len = int(X_len)    
        shortest_path = hop_matrices[idx, :X_len, :X_len]
        np.fill_diagonal(shortest_path, np.nan)
        total += np.nansum(shortest_path)
        n_tokens += X_len * (X_len - 1)
        # print(f'total = {total}, n_tokens = {n_tokens}')
    mean_shortest_paths.append(total / n_tokens)

    ##### Save #####
    if is_fns:
        alpha, bandwidth, a = attn_setup['alpha'], attn_setup['bandwidth'], attn_setup['a']
        fns_type = attn_setup['model_name']
        spectrum_file = f'attn_graph-{fns_type}-seed={seed}-alpha={alpha}-pretrained_embd={pretrained_model_name}-same_token={args.use_same_token}.npz'
        eval_file = f'evaluate-{fns_type}-seed={seed}-alpha={alpha}-pretrained_embd={pretrained_model_name}.npz'
    else:
        spectrum_file = f'attn_graph-dpformer-seed={seed}.npz'
        eval_file = f'evaluate-dpformer-seed={seed}.npz'
    # ii = 0
    # model_dir_split = args.model_dir.split('/')
    # while ii < len(model_dir_split):
    #     ss = model_dir_split[ii]
    #     if ss == '.droot':
    #         break
    #     ii += 1
    # SAVE_DIR = njoin(DROOT, 'pretrained_data', model_dir_split[ii+1], 
    #                  model_dir_split[ii+2])
    # if not isdir(SAVE_DIR): makedirs(SAVE_DIR)       

    # njoin(SAVE_DIR, spectrum_file)

    # ----- file 1 -----
    np.savez(njoin(args.model_dir, spectrum_file), spectrum=spectrum, left_eigvecs=top_left_eigvecs, 
             dist_matrices=dist_matrices, hop_matrices=hop_matrices, all_weights=all_weights, X_lens=X_lens)

    # ----- file 2 -----
    np.savez(njoin(args.model_dir, "andrew_results_1.npz"), attn_weights=attn_weights, 
             shortest_path=shortest_path, sequence=X) # Check that sequence = X would work?              

    # ----- file 3 -----             
    np.savez(njoin(args.model_dir, "andrew_results_2.npz"), mean_spectral_gaps=np.array(mean_spectral_gaps), 
             mean_shortest_paths=np.array(mean_shortest_paths)) 