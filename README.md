# Fractional neural attention for processing long sequences

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

## translation-final

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

[2] `long-range-arena` based on https://github.com/state-spaces/s4

[3] `nmt` based on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers

[4] `translation-final` based on https://github.com/tanjeffreyz/attention-is-all-you-need

[5] NanoGPT: https://github.com/karpathy/nanoGPT