import numpy as np
from UTILS.mutils import njoin, AttrDict, load_model_files
from constants import MODEL_SUFFIX, DROOT
from tokenization import PretrainedTokenizer
from UTILS.data_utils import create_examples
from torch.utils.data import DataLoader

def load_results(model_dir):
    model_dir = model_dir.replace('\\','')
    attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
    seed = attn_setup ['seed']
    fix_embed = attn_setup['fix_embed']
    pretrained_model_name = config['pretrained_model_name'] if fix_embed else False
    is_fns = attn_setup['model_name'][-9:] == 'fns' + MODEL_SUFFIX 
    if is_fns:
        alpha, bandwidth, a = attn_setup['alpha'], attn_setup['bandwidth'], attn_setup['a']
        fns_type = attn_setup['model_name']
        spectrum_file = f'attn_graph-{fns_type}-seed={seed}-alpha={alpha}-pretrained_embd={pretrained_model_name}-same_token={use_same_token}.npz'
        eval_file = f'evaluate-{fns_type}-seed={seed}-alpha={alpha}-pretrained_embd={pretrained_model_name}.npz'
    else:
        spectrum_file = f'attn_graph-dpformer-seed={seed}.npz'
        eval_file = f'evaluate-dpformer-seed={seed}.npz'

    SAVE_DIR = njoin(DROOT, 'pretrained_data', model_dir.split('/')[1],
                    model_dir.split('/')[2])
    results = np.load(njoin(SAVE_DIR, spectrum_file))
    return results

if __name__ == '__main__':
    
    # Add model directories for alpha = 1.2, 2, DPformer with some fixed dimension (maybe just the same dimension as the one for diffusion map result previously)
    model_dirs = [...]
    use_same_token = True
    
    attn_weights = np.zeros((3, 512, 512))
    shortest_path = np.zeros((3, 512, 512))
    for model_idx, model_dir in enumerate(model_dirs):
        results = load_results(model_dir)
        # Get results for first full-length sequence
        X_lens = results['X_lens']
        idx = np.where(X_lens == X_lens.max())[0]
        attn_weights[model_idx, :, :] = results['all_weights'][idx, :, :] # Attention weights
        shortest_path[model_idx, :, :] = results['hop_matrices'][idx, :, :] # Shortest path
        if model_idx == 0: # Load dataset - has the seed for dataset been set? will we get data in different order?
            attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
            tokenizer = PretrainedTokenizer(pretrained_model='wiki.model', vocab_file='wiki.vocab')
            config['dataset_name'] = attn_setup['dataset_name']
            config['max_len'] = config['seq_len']
            main_args = AttrDict(config)  
            train_dataset = create_examples(main_args, tokenizer, mode='train')
            test_dataset = create_examples(main_args, tokenizer, mode='test')                    
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)  
            # Figure out what that sequence was...
            np.random.seed(0)
            N_sample = 100
            sample_idxs = np.random.choice(len(train_dataset), N_sample, replace=False)
            seq_idx = sample_idxs[idx] # Sequence index
            X, Y = train_dataset[seq_idx] # Should be the corresponding sequence
    # Save results
    np.savez("andrew_results_1.npz", attn_weights=attn_weights, shortest_path=shortest_path, sequence=X) # Check that sequence = X would work?

    # Loop through all available models
    all_model_dirs = [...]
    use_same_token = True
    mean_spectral_gaps = []
    mean_shortest_paths = []
    for model_dir in all_model_dirs:
        results = load_results(model_dir)
        X_lens = results['X_lens']
        spectrum = results['spectrum']
        # Find mean spectral gap for each model
        spectrum = np.sort(spectrum, axis=-1)
        spectral_gap = spectrum[:, -1] - spectrum[:, -2]
        mean_spectral_gap = np.mean(spectral_gap)
        mean_spectral_gaps.append(mean_spectral_gap)
        # Find mean shortest path
        total = 0
        n_tokens = 0
        for idx, X_len in enumerate(X_lens):
            shortest_path = results['hop_matrices'][idx, :X_len, :X_len]
            total += np.nansum(np.fill_diagonal(shortest_path, np.nan))
            n_tokens += X_len * (X_len - 1)
        mean_shortest_paths.append(total / n_tokens)
    # Save results
    np.savez("andrew_results_2.npz", mean_spectral_gaps=np.array(mean_spectral_gaps), mean_shortest_paths=np.array(mean_shortest_paths)) 