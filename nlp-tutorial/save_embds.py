import json
import numpy as np
import os
import torch

from os import makedirs
from os.path import isfile, isdir
from tqdm import tqdm
from utils.mutils import njoin, load_model_files
from constants import DROOT

data_path = njoin(DROOT, 'depth_analysis')
save_path = njoin(DROOT, 'saved_embeddings')

for layer_dir in tqdm(os.listdir(data_path)):
    save_layer_dir = njoin(save_path, layer_dir)
    layer_dir = njoin(data_path, layer_dir)    
    for qk_dir in os.listdir(layer_dir):
        if 'config_' in qk_dir:
            save_qk_dir = njoin(save_layer_dir, qk_dir)
            qk_dir = njoin(layer_dir, qk_dir)            
            for dataset_dir in os.listdir(qk_dir):
                save_dataset_dir = njoin(save_qk_dir, dataset_dir)
                dataset_dir = njoin(qk_dir, dataset_dir)                
                for config_dir in os.listdir(dataset_dir):
                    save_config_dir = njoin(save_dataset_dir, config_dir)
                    config_dir = njoin(dataset_dir, config_dir)                    
                    for model_dir in os.listdir(config_dir):
                        save_model_dir = njoin(save_config_dir, model_dir)
                        model_dir = njoin(config_dir, model_dir)                        
                        for seed_dir in os.listdir(model_dir):
                            save_seed_dir = njoin(save_model_dir, seed_dir)
                            seed_dir = njoin(model_dir, seed_dir)                            

                            # load weights
                            f = open(njoin(seed_dir, 'config.json'))
                            config = json.load(f)
                            checkpoint = njoin(seed_dir, 'ckpt.pt')
                            ckpt = torch.load(checkpoint)

                            if 'rdfnsformer' in seed_dir:
                                from models.rdfnsformer import RDFNSformer
                                model = RDFNSformer(config)
                            elif 'dpfnsformer' in seed_dir:
                                from models.dpformer import DPformer
                                model = DPformer(config)                                

                            embds = ckpt['model']['embedding.weight'].cpu().detach().numpy()
                            #pos_embds = ckpt['model']['pos_embedding.weight'].cpu().detach().numpy()
                            pos_embds = model.pos_embedding.weight.cpu().detach().numpy()
                            makedirs(save_seed_dir, exist_ok=True)                            
                            np.save(njoin(save_seed_dir, 'embds'), embds)
                            np.save(njoin(save_seed_dir, 'pos_embds'), pos_embds)

                            # test code
                            # loaded_embds = np.load(njoin(save_seed_dir, 'embds.npy'))
                            # loaded_pos_embds = np.load(njoin(save_seed_dir, 'pos_embds.npy'))

