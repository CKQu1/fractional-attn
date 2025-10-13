# Fractional neural attention for processing long sequences

## nlp-tutorial

The script `batch_exps.py` contains all the experiments ran in the manuscript, i.e. including `exps1`, `exps2`, `exps3`. These experiemtns can be submitted to the cluster through `batch_submit_main.py`. Otherwise, the single jobs could be ran through `main.py`.

## vit-pytorch

A single instance of training can be realized through:
`python main.py`

For batch submitting all the training instances in the manuscript:
`python batch_submit_main.py`

## long-range-arena

### Data preparation

Run `lra_dl.sh` or submit job via `qsub lra_dl.sh`, then execute python `aan_script.py` to download datasets.

### Train models

We focus on 2 types of experiments:
    1. `python main.py --dataset_name=aan-classification`
    2. `python main.py --dataset_name=pathfinder-classification`

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

### Batch job submission

Run `python batch_submit_main.py`

## Code References

[1] `nlp-tutorial`, `vit-pytorch` based on https://github.com/michaelsdr/sinkformers/

[2] `long-range-arena` based on https://github.com/state-spaces/s4

[3] `nmt` based on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers

[4] `translation-final` based on https://github.com/tanjeffreyz/attention-is-all-you-need

[5] NanoGPT: https://github.com/karpathy/nanoGPT