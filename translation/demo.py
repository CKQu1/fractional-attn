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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main.py training arguments')  
    parser.add_argument('--model_dir', default='', type=str, help='model seed dir')
    parser.add_argument('--is_train', type=str2bool, default=False)
    parser.add_argument('--didx', default=0, type=int)

    args = parser.parse_args()

    model_dir = args.model_dir
    
    # Download and preprocess data
    CONFIG.BATCH_SIZE = 1  # reset batch size to 1
    dataset = Dataset(CONFIG.LANGUAGE_PAIR, batch_size=CONFIG.BATCH_SIZE)

    # load weights to model
    f = open(njoin(model_dir,'config.json'))
    config = json.load(f)
    model = Transformer(config)
    device = model.device
    checkpoint_parent = njoin(model_dir, 'weights')
    checkpoint = njoin(checkpoint_parent, os.listdir(checkpoint_parent)[0])
    ckpt = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(ckpt)        

    if args.is_train:
        data = next(iter(dataset.train_loader))
    else:
        data = next(iter(dataset.valid_loader))
    src = data['source'].to(model.device)
    trg = data['target'].to(model.device)

    predictions = model(src, trg[:, :-1])

    # Calculate BLEU score
    batch_size = predictions.size(0)
    batch_bleu = 0
    p_indices = torch.argmax(predictions, dim=-1)
    i = 0
    p_tokens = dataset.trg_vocab.lookup_tokens(p_indices[i].tolist())
    t_tokens = dataset.trg_vocab.lookup_tokens(trg[i, 1:].tolist())

    # Filter out special tokens
    p_tokens = list(filter(lambda x: '<' not in x, p_tokens))
    t_tokens = list(filter(lambda x: '<' not in x, t_tokens))

    if len(p_tokens) > 0 and len(t_tokens) > 0:
        batch_bleu += sentence_bleu([t_tokens], p_tokens)
        #batch_bleu += sacrebleu.corpus_bleu([t_tokens], p_tokens).score

    print(f'Network output: {p_tokens} \n')

    print(f'Desired translation: {t_tokens} \n')