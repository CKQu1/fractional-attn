# Fractional neural attention for processing long sequences

## nlp-tutorial

The script `batch_exps.py` contains all the experiments ran in the manuscript, i.e. including `exps1`, `exps2`, `exps3`. These experiemtns can be submitted to the cluster through `batch_submit_main.py`. Otherwise, the single jobs could be ran through `main.py`.

## vit-pytorch

Add later...

## long-range-arena

### Data preparation

Run `lra_dl.sh` or submit job via `qsub lra_dl.sh`, then execute python `aan_script.py` to download datasets.

### Train models

We focus on 2 types of experiments:
    1. `python main.py --dataset_name=aan-classification`
    2. `python main.py --dataset_name=pathfinder-classification`

## nmt

### Data preparation

Run `python prepare_data.py`

### Model training

Run `python main.py`

## Code References

[1] `nlp-tutorial`, `vit-pytorch` based on https://github.com/michaelsdr/sinkformers/

[2] `long-range-arena` based on https://github.com/state-spaces/s4

[3] `nmt` based on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers

[4] NanoGPT: https://github.com/karpathy/nanoGPT