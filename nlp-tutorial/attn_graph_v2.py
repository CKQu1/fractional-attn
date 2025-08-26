import argparse
import os
from os import makedirs
from os.path import isdir
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase
import numpy as np
import torch
from torch.nn import functional as F  
from utils.mutils import njoin, str2bool, collect_model_dirs, AttrDict, load_model_files, dist_to_score
from constants import HYP_CMAP, HYP_CNORM, FIGS_DIR, MODEL_SUFFIX
from models.rdfnsformer import RDFNSformer
from models.dpformer import DPformer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from utils.data_utils import glove_create_examples
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx
import torch.nn as nn

# ----- Global Variables -----
MARKERSIZE = 4
BIGGER_SIZE = 10
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def count_trailing_zeros(tensor):
    # Reverse the tensor and find the first non-zero element
    reversed_tensor = torch.flip(tensor, dims=[0])
    nonzero_indices = torch.nonzero(reversed_tensor, as_tuple=True)[0]
    
    if len(nonzero_indices) == 0:  # All elements are zero
        return len(tensor)
    else:  # Count zeros from the end
        return nonzero_indices[0].item()
    
def eval_model(model):
    _loss = 0
    _acc = 0
    corrects = []
    lengths = []
    for batch in test_dataset:
        inputs, labels = batch
        
        inputs = inputs.to(device)
        labels = labels.to(device)            

        if model_idx == 0: # opdpformer
            outputs, _ = model(inputs[None])
        else:
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

    # Training options
    parser = argparse.ArgumentParser(description='nlp-tutorial/fdm.py arguments')    
    # parser.add_argument('--train_with_ddp', default=False, type=bool, help='to use DDP or not')
    parser.add_argument('--models_root', default='', help='Pretrained models root')
    parser.add_argument('--fns_type', default='opv2_rdfnsformer')  # 'spopfns'+MODEL_SUFFIX
    parser.add_argument('--is_use_mask', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_same_token', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--layers', type=int, default=8, help='Number of layers')

    parser.add_argument('--is_3d', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_log', default=False, type=bool)

    args = parser.parse_args()   
    
    # ----- set up -----
    device = "cpu"
    loss_fn = nn.CrossEntropyLoss()

    # Get model setting from dir
    models_root = args.models_root.replace('\\','')
    dirnames = sorted([dirname for dirname in os.listdir(models_root) if 'former' in dirname])  

    # select based on bandwidth    
    DCT_ALL = collect_model_dirs(models_root, suffix=MODEL_SUFFIX)
    model_types = list(DCT_ALL.keys())
    dp_types = [model_type for model_type in model_types if 'dp' in model_type]

    #SELECTED_ALPHAS = None
    SELECTED_ALPHAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # SELECTED_ALPHAS = [2.0,]
    #EXCLUDED_EPSS = None
    EXCLUDED_EPSS = [100, 10, 0.1, 0.01, 0.001]
    for model_type in model_types:
        if args.fns_type in model_type:            
            df_model = DCT_ALL[model_type].dropna(subset='alpha')
            if SELECTED_ALPHAS is not None:
                # ----- filter alphas -----
                df_model = df_model[df_model['alpha'].isin(SELECTED_ALPHAS)]
                # ------------------------
                df_model.reset_index(drop=True, inplace=True)
            if EXCLUDED_EPSS is not None:
                # ----- filter alphas -----
                df_model = df_model[~df_model['bandwidth'].isin(EXCLUDED_EPSS)]
                # ------------------------
                df_model.reset_index(drop=True, inplace=True)
            break
        
    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())  # small to large
    epss = sorted(df_model.loc[:,'bandwidth'].unique())  
    
    # Figure set up
    nrows, ncols = 5, len(alphas)
    figsize = (3*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                            sharex=False,sharey=False)  
    if nrows == 1:
       axs = np.expand_dims(axs, axis=0) if ncols > 1 else np.expand_dims(axs, axis=[0,1])
       
    # total_figs = 0        
    # for row in range(nrows):
    #     for col in range(ncols):
        #     axs[row,col].text(
        # 0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
        #     axs[row,col].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        # va='bottom')
            # total_figs += 1
            
    for eps_idx, eps in enumerate(epss):

        model_dirs = []
        for ii in range(df_model.shape[0]):
            ensembles = df_model.loc[ii,'ensembles']            
            if ensembles > 0 and df_model.loc[ii,'bandwidth'] == eps:
                seed = str(sorted(df_model.loc[ii,'seeds'])[-1])
                model_dirs.append(njoin(df_model.loc[ii,'model_dir'], f'model={seed}'))

                #attn_setup, config, _, _ = load_model_files(model_dir)
        
        # Include opdpformer
        model_dirs.append(njoin(DCT_ALL['opdpformer'].loc[0,'model_dir'], f'model={seed}'))
        
        print(f'Bandwidth: {eps}, seed selected: {seed}')

        models = []  # delete
        model_dirs = sorted(model_dirs)
        for model_idx, model_dir in enumerate(model_dirs):

            # ----- load pretrained_model -----    
            attn_setup, config, run_performance, train_setting = load_model_files(model_dir)

            fix_embed = attn_setup['fix_embed']
            pretrained_model_name = config['pretrained_model_name'] if fix_embed else False

            config['dataset_name'] = attn_setup['dataset_name']
            config['max_len'] = config['seq_len']
            # manifold = attn_setup['manifold']
            # if 'is_rescale_dist' not in config.keys():
            #     # maybe add prompt
            #     config['is_rescale_dist'] = args.is_rescale_dist  # manual setup            

            main_args = AttrDict(config)

            if model_idx == 0: # opdpformer
                model = DPformer(config)
            else: # rdfnsformer
                model = RDFNSformer(config, is_return_dist=True)     
            checkpoint = njoin(model_dir, 'ckpt.pt')
            ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt['model'])
            model.eval()
            models.append(model)  # delete

            # load dataset
            if model_idx == 0:                
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
                        from utils.data_utils import create_examples
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
                        from utils.data_utils import create_examples
                        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                        pretrained_model = AlbertModel.from_pretrained("albert-base-v2")                
                        vocab_size, pretrained_model_hidden =\
                            pretrained_model.embeddings.word_embeddings.weight.shape
                        pretrained_seq_len, _ = pretrained_model.embeddings.position_embeddings.weight.shape

                        train_dataset = create_examples(main_args, tokenizer, mode='train')
                        test_dataset = create_examples(main_args, tokenizer, mode='test')    

                else:
                    from tokenization import Tokenizer, PretrainedTokenizer
                    from utils.data_utils import create_examples
                    tokenizer = PretrainedTokenizer(pretrained_model='wiki.model', vocab_file='wiki.vocab')

                    train_dataset = create_examples(main_args, tokenizer, mode='train')
                    test_dataset = create_examples(main_args, tokenizer, mode='test')                    

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  
            
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
                if model_idx == 0: # opdpformer
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
            
            ##### Save #####
            if model_idx > 0:
                alpha, bandwidth, a = attn_setup['alpha'], attn_setup['bandwidth'], attn_setup['a']
                spectrum_file = f'attn_graph-{args.fns_type}-eps={eps}-pretrained_embd={pretrained_model_name}-same_token={args.use_same_token}-alpha={alpha}_{args.layers}.npz'
                eval_file = f'evaluate-{args.fns_type}-eps={eps}-pretrained_embd={pretrained_model_name}-alpha={alpha}_{args.layers}.npz'
            else:
                spectrum_file = f'attn_graph-dpformer_{args.layers}.npz'
                eval_file = f'evaluate-dpformer_{args.layers}.npz'
            if 'L-hidden' in args.models_root.split('/')[1]:
                SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1])
            else:
                SAVE_DIR = njoin(FIGS_DIR, 'pretrained_analysis', args.models_root.split('/')[1], 
                                args.models_root.split('/')[2])
            if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    

            if not isdir(SAVE_DIR): makedirs(SAVE_DIR)               
            np.savez(njoin(SAVE_DIR, spectrum_file), spectrum=spectrum, left_eigvecs=top_left_eigvecs, dist_matrices=dist_matrices, hop_matrices=hop_matrices, all_weights=all_weights, X_lens=X_lens)