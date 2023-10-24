import os
import sys
from os.path import join, normpath
sys.path.append(os.getcwd())
from path_setup import droot    


def get_dims():
    import dgl
    import torch
    import numpy as np
    import os
    import uuid

    from os.path import join
    from sklearn.metrics import f1_score
    from transformers import TrainingArguments, DataCollatorWithPadding
    from transformers import RobertaTokenizer
    from transformers.utils import logging
    from datasets import load_dataset,load_metric,load_from_disk
    from models.diffuser_app import DiffuserForSequenceClassification
    from models.diffuser_utils import DiffuserConfig
    from graphtrainer import graphTrainer

    dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                    if torch.cuda.is_available() else "cpu")

    logging.set_verbosity_debug()
    logger = logging.get_logger()

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding = 'max_length', truncation=True, max_length = 1024)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        f1_score = metric_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": acc, "f1_score": f1_score }

    metric_acc = load_metric('./metrics/accuracy')
    metric_f1 = load_metric('./metrics/f1')

    # create cache for dataset
    dataset_dir = join(droot, "DATASETS")
    if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)

    imdb = load_dataset("imdb", cache_dir=dataset_dir)
    # tokenizer = RobertaTokenizer.from_pretrained("./roberta-tokenizer", max_length = 1024)
    tokenizer = RobertaTokenizer(tokenizer_file = "./roberta-tokenizer/tokenizer.json",
                                vocab_file     = "./roberta-tokenizer/vocab.json",
                                merges_file    = "./roberta-tokenizer/merges.txt",
                                max_length     = 1024)

    # save tokenized dataset
    dataset_dir = join(droot, "DATASETS", "tokenized_imdb")
    if not os.path.isdir(dataset_dir): 
        print("Downloading data!")
        tokenized_imdb = imdb.map(preprocess_function, batched=True)
        tokenized_imdb = tokenized_imdb.map(remove_columns=["text"])
        os.makedirs(dataset_dir)
        tokenized_imdb.save_to_disk(dataset_dir)
    else:
        print("Data downloaded, loading from local now!")
        tokenized_imdb = load_from_disk(dataset_dir)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    config = DiffuserConfig.from_json_file("./models/config_simple.json")
    config.num_labels = 2
    with_frac = False

    model =  DiffuserForSequenceClassification(config=config, with_frac=with_frac).to(dev)

    uuid_ = str(uuid.uuid4())[:8]
    training_args = TrainingArguments(
        output_dir = join(droot, "trained", "save_imdb",uuid_),
        #output_dir = None,
        learning_rate = 3e-5,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        #num_train_epochs = 1,
        num_train_epochs = 0.0025,
        weight_decay = 0.01,
        evaluation_strategy = "steps",
        eval_steps = 2,
        #logging_steps = 500,
        #save_steps = 500,
        logging_steps = 0.2,
        save_steps = 0.2,    
        seed = 42,
        #warmup_steps = 50,
        warmup_steps = 2,
        gradient_accumulation_steps = 8,
        prediction_loss_only=True
    )

    if dev.type != "cpu":
        steps_per_train_epoch       = int(len(tokenized_imdb['train'])/(training_args.per_device_train_batch_size*torch.cuda.device_count()*training_args.gradient_accumulation_steps ))
    else:
        steps_per_train_epoch       = int(len(tokenized_imdb['train'])/(training_args.per_device_train_batch_size*torch.get_num_threads()*training_args.gradient_accumulation_steps ))
    training_args.eval_steps    = int(steps_per_train_epoch)
    training_args.logging_steps = int(steps_per_train_epoch/5)
    training_args.save_steps    = int(steps_per_train_epoch)

    trainer = graphTrainer(
        model = model,
        config = config,
        args = training_args,
        train_dataset = tokenized_imdb["train"],
        eval_dataset = tokenized_imdb["test"],
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )    


    ## 0.2 Sparse pattern design ------------------------------------------------------------
    print("0.2 Sparse pattern design \n")

    attention_window = (
        trainer.config.attention_window
        if isinstance(trainer.config.attention_window, int)
        else max(trainer.config.attention_window)
    )

    #max_len = 4096 # not the input sequence max len
    max_len = 1024 # not the input sequence max len
    n_blocks = max_len//(attention_window//2)-1        

    return max_len, attention_window, n_blocks, max_len*trainer.config.num_rand, trainer.config.num_glob


def get_full_pattern():
    import numpy as np

    max_len, attention_window, n_blocks, num_random, num_global = get_dims()
    print(f"max_len: {max_len}, attention_window: {attention_window}, n_blocks: {n_blocks} \n")

    adj = np.zeros([max_len, max_len])
    # add local window att (overlap)
    for i in range(n_blocks):
        start = i*attention_window//2
        end = start+attention_window
        if end > max_len:
            end = max_len
        adj[start:end, start:end] = 1

    # add random att    
    np.random.seed(0)    
    idx = np.random.choice(range(max_len*max_len), num_random ,replace=False)
    idx_x = idx %  max_len
    idx_y = idx // max_len
    adj[idx_x,idx_y] = 1

    # add global att        
    idx = np.random.choice(range(attention_window,max_len), num_global ,replace=False)
    adj[idx,:] = 1
    adj[:,idx] = 1

    possible_seq_len = np.arange(attention_window, max_len+attention_window, attention_window)
    src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}    

    return adj, possible_seq_len, src_dst    


def save_sparse_pattern():
    import numpy as np
    from os.path import join

    max_len, attention_window, n_blocks, num_random, num_global = get_dims()
    print(f"max_len: {max_len}, attention_window: {attention_window}, n_blocks: {n_blocks} \n")

    adj = np.zeros([max_len, max_len])
    # add local window att (overlap)
    for i in range(n_blocks):
        start = i*attention_window//2
        end = start+attention_window
        if end > max_len:
            end = max_len
        adj[start:end, start:end] = 1

    data_dir = join(droot, "sparse_pattern")
    if not os.path.isdir(data_dir): os.makedirs(data_dir)

    np.save(join(data_dir, "local_pattern"), adj)
    adj = np.zeros([max_len, max_len])

    # add random att    
    np.random.seed(0)    

    idx = np.random.choice(range(max_len*max_len), num_random ,replace=False)
    idx_x = idx %  max_len
    idx_y = idx // max_len
    adj[idx_x,idx_y] = 1

    np.save(join(data_dir, "random_pattern"), adj)
    adj = np.zeros([max_len, max_len])

    # add global att    
    idx = np.random.choice(range(attention_window,max_len), num_global ,replace=False)
    adj[idx,:] = 1
    adj[:,idx] = 1

    np.save(join(data_dir, "global_pattern"), adj)

    print("Sparse patterns saved!")

    possible_seq_len = np.arange(attention_window, max_len+attention_window, attention_window)
    src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}    

    return possible_seq_len, src_dst

def plot_pattern():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    global adj, data_dir, pattern_types, pattern_type
    global possible_seq_len, src_dst

    # get patterns
    possible_seq_len, src_dst = save_sparse_pattern()

    pattern_types = ["local_pattern", "random_pattern", "global_pattern"]

    data_dir = join(droot, "sparse_pattern")
    if not os.path.isdir(data_dir): os.makedirs(data_dir)

    alphas = [1, 0.65, 0.3]
    colors = ["red", "blue", "orange"]
    cmaps = []
    norms = []
    for idx, pattern_type in enumerate(pattern_types):
        
        adj = np.load(normpath(join(data_dir, f"{pattern_type}.npy")))

        c_ls = ['none', colors[idx]]
        bounds = [0,1]
        cmap = mpl.colors.ListedColormap(c_ls)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cmaps.append(cmap); norms.append(norm)

        img = plt.imshow(adj, interpolation='none', cmap=cmaps[idx], norm=norms[idx])
        #plt.plot([], [], c=colors[idx], alpha=alphas[idx], label=pattern_type)
        plt.plot([], [], c=colors[idx], alpha=1, label=pattern_type)

        #break

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    #plt.show()
    plt.savefig(join(droot, "sparse_pattern", "sparsify_cmap.pdf"))

# for verifying same function in graphtrainer.py
def _from_adj_to_batched_sparsify(B, seq_len):
    import matplotlib.pyplot as plt
    import os
    import torch
    from os.path import join
    sys.path.append(os.getcwd())
    from path_setup import droot    
    global sparsify, batched_sparsify, src, dst
    B, seq_len = int(B), int(seq_len)
    adj, possible_seq_len, src_dst = get_full_pattern()
    src,dst = src_dst[seq_len]
    sparsify = torch.zeros([seq_len, seq_len])
    sparsify[src, dst] = 1
    batched_sparsify = sparsify.repeat(B, 1, 1)
    plt.imshow(sparsify, cmap="Greys")
    plt.savefig(join(droot, "sparse_pattern", "sparsify.pdf"))
    return batched_sparsify         

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])