
# Fractional attention sampling

  
We introduce a fractional diffusion approach to the attention graph that converges to a fractional graph Laplacian in the asymptotic limit. This facilitates a power-law decay in the density of token nodes located at a distance, thus enabling efficient preservation of memory.

## Code
Important files in the code are:
*   [diffuser_att.py](./models/diffuser_att.py): The core attention module
*   [diffuser.py](./models/diffuser.py): The Diffuser layer architecture
*   [diffuser_app.py](./models/diffuser_app.py): Application wrappers for different tasks
*   [graphtrainer.py](./graphtrainer.py): Customized trainer defining sparse patterns as graphs

### Installation 
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.8.0 version.

The other important dependencies requirements are listed in [requirements.txt](./requirements.txt).

- `conda create --name diffuser python=3.8`
- `conda activate diffuser`
- `pip install -r requirements.txt`
- `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch` (https://pytorch.org/get-started/previous-versions/)
    - `pytorch==1.8.0` is not compatible with DGL
    - error message: `RuntimeError: DGL requires PyTorch >= 1.12.0`
- `pip install dgl -f https://data.dgl.ai/wheels-test/repo.html   # or dgl-cuXX for CUDA` (https://github.com/dmlc/dgl/releases) 
- or `pip install dgl_cu113 -f https://data.dgl.ai/wheels-test/repo.html` (doesn't work actually)
- or `conda install -c dglteam dgl-cuda10.2==0.8.2.post1` (this seems to work)
    - https://blog.csdn.net/qq_30049011/article/details/120763171
    - https://conda.anaconda.org/dglteam/linux-64
    - https://stackoverflow.com/questions/76519346/unable-to-install-dgl-cuany-version-in-google-colab
- or use pip install which seems to work
    ```
    !pip install  dgl -f https://data.dgl.ai/wheels/cu102/repo.html
    !pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
    ```

### Running Classification
To run IMDB review classification task with one GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train_classification_imdb.py 
```
Multi-GPU training has to be lauched with DistributedDataParallel (DDP) for PyTorch and DGL
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_classification_imdb.py
 ```
Model configurations are listed in [config.json](./models/config.json) and training arguments can be changed in [train_classification_imdb.py](./train_classification_imdb.py)

To run general sequence classification tasks
    - `python main_seq_classification.py --train_with_ddp=True --lr=0.01` (as an example)
To qsub jobs
    - `python submit_main.py`

To get sparsification pattern saved as figure:
    - python dynamics_study/sparse_pattern_design.py plot_pattern


## References
****
Please refer to the manuscript.


## Github references
****
[1] Diffuser: Efficient Transformers with Multi-hop Attention Diffusion for Long Sequences, https://github.com/asFeng/Diffuser
