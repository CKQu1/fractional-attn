# Fractional neural attention for processing long sequences

## Environment information

I used `Python 3.11.13` throughout. For `nlp-tutorial` and `vit-pytorch`, the requirements are as in `requirements_fsa.txt`; for `translation` in `requirements_nmt.txt`.

## nlp-tutorial

The script `batch_exps.py` contains all the experiments ran in the manuscript, i.e. including `exps1`, `exps2`, `exps3`. These experiemtns can be submitted to the cluster through `batch_submit_main.py`. Otherwise, the single jobs could be ran through `main.py`.

### Experiments

1. Training models of 6 layers and 8 attention heads (Euclidean FNA):
    - `python batch_submit_main.py --exp=exp1`
2. Training models of single-layers and various embedding dimension:
    - `python batch_submit_main.py --exp=exp2`
3. Dynamic inference for single-layer models:
    - `python batch_submit_main.py --exp=exp3`
4. Spectral gap analysis for single-layer models:
    - `python batch_submit_main.py --exp=exp5`
5. Attention graph analysis for single-layer models:
    - `python batch_submit_main.py --exp=exp6`
6. Training models of 6 layers and 8 attention heads (spherical FNA):
    - `python batch_submit_main.py --exp=exp7`

## vit-pytorch

A single instance of training can be realized through:
`python main.py`

### Experiments

Training models in the paper:
`python batch_submit_main.py`

## translation

### Data preparation

`python -m spacy download en_core_web_sm`
`python -m spacy download de_core_news_sm`

### Train model

`python --model_name=fnsformer --alpha=1.2 --a=0 --bandwidth=1 --manifold=rd --is_rescale_dist=True --model_root=.droot/experiments --seed=0 --is_op=False --lr=0.00022 --lr_reduction_factor=0.75`

### Files to check

Double-check `modules/attention.py`
    - Can use `tests/model.py` to check things too
    - Can use `demo.py` to check translation output of model

### Experiments

Run `python batch_submit_main.py`

## Code References

[1] `nlp-tutorial`, `vit-pytorch` based on https://github.com/michaelsdr/sinkformers/

[2] `translation` based on https://github.com/tanjeffreyz/attention-is-all-you-need

[3] NanoGPT: https://github.com/karpathy/nanoGPT

## Cite Our Paper

Preprint link: https://arxiv.org/abs/2511.10208 

@ARTICLE{fna_model,
       author = {{Qu}, Cheng Kevin and {Ly}, Andrew and {Gong}, Pulin},
        title = "{Fractional neural attention for efficient multiscale sequence processing}",
      journal = {arXiv e-prints},
     keywords = {Machine Learning, Artificial Intelligence, Dynamical Systems, Probability, Biological Physics},
         year = 2025,
        month = nov,
          eid = {arXiv:2511.10208},
        pages = {arXiv:2511.10208},
          doi = {10.48550/arXiv.2511.10208},
archivePrefix = {arXiv},
       eprint = {2511.10208},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv251110208Q},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}