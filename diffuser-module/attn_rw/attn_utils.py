import dgl
import dgl.function as fn
import math
import os
import sys
import torch

sys.path.append(f'{os.getcwd()}')
from path_setup import droot

from os.path import join
from functools import reduce
from dgl.nn.functional import edge_softmax
#from models.diffuser_utils import *
#from models.utils import *
from torch import nn
from time import time
from tqdm import tqdm

dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")
print(f"Device used: {dev}")

## 0.1 Import transformer ------------------------------------------------------------
print("0.1 Import transformer \n")
repo_dir = os.getcwd()
print(repo_dir)

from sklearn.metrics import f1_score
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import RobertaTokenizer
from transformers.utils import logging
from datasets import load_dataset,load_metric,load_from_disk
from models.diffuser_app import DiffuserForSequenceClassification
from models.diffuser_utils import DiffuserConfig
from graphtrainer import graphTrainer

#logging.set_verbosity_debug()
#logger = logging.get_logger()

def get_hidden_states(dataset_name, max_length=1024, use_dgl=True, config_path=f"{repo_dir}/models/config_simple.json"):

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding = 'max_length', truncation=True, max_length = max_length)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        f1_score = metric_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": acc, "f1_score": f1_score }

    metric_acc = load_metric(f'{repo_dir}/metrics/accuracy')
    metric_f1 = load_metric(f'{repo_dir}/metrics/f1')

    # create cache for dataset
    dataset_dir = join(droot, "DATASETS")
    if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)

    if dataset_name == "imdb":
        dataset = load_dataset(dataset_name, cache_dir=dataset_dir)
    else:
        print("Can only allow imdb as a dataset for now!")
        quit()

    #tokenizer = RobertaTokenizer.from_pretrained("./roberta-tokenizer", max_length = max_length)
    tokenizer = RobertaTokenizer(tokenizer_file = f"{repo_dir}/roberta-tokenizer/tokenizer.json",
                                 vocab_file     = f"{repo_dir}/roberta-tokenizer/vocab.json",
                                 merges_file    = f"{repo_dir}/roberta-tokenizer/merges.txt",
                                 max_length     = max_length)

    # save tokenized dataset
    dataset_dir = join(droot, "DATASETS", f"tokenized_{dataset_name}")
    if not os.path.isdir(dataset_dir): 
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.map(remove_columns=["text"])
        os.makedirs(dataset_dir)
        tokenized_dataset.save_to_disk(dataset_dir)
    else:
        tokenized_dataset = load_from_disk(dataset_dir)
    # convert to torch (doesn't work)
    #tokenized_dataset.with_format('torch')     # device=dev, not rlly needed since this is already done in graphtrainer.py
    #dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=5000)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    config = DiffuserConfig.from_json_file(config_path)
    config.num_labels = 2
    with_frac = False
    model =  DiffuserForSequenceClassification(config=config, with_frac=with_frac).to(dev)

    # dummy training_args
    training_args = TrainingArguments(        
        output_dir = join(droot, "attn_pass"),
        learning_rate = 3e-5,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        num_train_epochs = 1,
        weight_decay = 0.01,
        evaluation_strategy = "steps",
        eval_steps = 50,
        logging_steps = 500,
        save_steps = 500,
        seed = 42,
        warmup_steps = 10,
        gradient_accumulation_steps = 8,
        prediction_loss_only=True
    )

    if dev.type != "cpu":
        steps_per_train_epoch       = int(len(tokenized_dataset['train'])/(training_args.per_device_train_batch_size*torch.cuda.device_count()*training_args.gradient_accumulation_steps ))
    else:
        steps_per_train_epoch       = int(len(tokenized_dataset['train'])/(training_args.per_device_train_batch_size*torch.get_num_threads()*training_args.gradient_accumulation_steps ))
    training_args.eval_steps    = int(steps_per_train_epoch)
    training_args.logging_steps = int(steps_per_train_epoch/5)
    training_args.save_steps    = int(steps_per_train_epoch)

    trainer = graphTrainer(
        use_dgl = use_dgl,
        model = model,
        config = config,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["test"],
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )

    # take small batch size sample for test
    """
    trainset = {}
    batch_size = 10
    print(tokenized_dataset['train'].features.keys())
    for data_type in tokenized_dataset['train'].features.keys():
    if data_type != "label":
        trainset[data_type] = torch.tensor(tokenized_dataset['train'][data_type][:batch_size]).to(dev)
    """

    trainset = tokenized_dataset['train'].shard(num_shards=5000, index=0)
    dset = {}
    for data_type in trainset.features.keys():
        if data_type != "label":
            dset[data_type] = torch.tensor(trainset[data_type])
        
    print("Computing loss")    
    #loss, outputs = trainer.compute_loss(model, trainset)
    #loss, outputs = trainer.compute_loss(model, dset, True)
    #loss, outputs = trainer.compute_loss(model, trainset, True)

    ## 0.2 Attention graph construction ------------------------------------------------------------
    print("0.2 Attention graph construction \n")

    # source and destination nodes src_dst
    src_dst = trainer.src_dst

    # from graphtrainer.py
    def _pad_to_window_size(inputs):
        attention_window = (
            config.attention_window
            if isinstance(config.attention_window, int)
            else max(config.attention_window)
        )
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = torch.tensor(inputs["input_ids"]).shape if inputs["input_ids"] is not None else torch.tensor(inputs["attention_mask"]).shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logging.debug(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if inputs["input_ids"] is not None:
                inputs["input_ids"] = nn.functional.pad(inputs["input_ids"], (0, padding_len), value=config.pad_token_id)
            inputs["attention_mask"] = nn.functional.pad(
                inputs["attention_mask"], (0, padding_len), value=False
            )  # no attention on the padding tokens
        return inputs

    trainset = _pad_to_window_size(trainset)

    # from graphtrainer.py
    """
    def _from_adj_to_batched_graphs(input_ids):
        input_ids = torch.tensor(input_ids)     # added

        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        g_list = []
        for i in range(B):
            src,dst = src_dst[seq_len]  # src_dst from above
            g = dgl.graph((src, dst))
            g_list.append(g)
        batched_g = dgl.batch(g_list)
        return batched_g  

    #batched_g = _from_adj_to_batched_graphs(trainset['input_ids'])
    batched_g = _from_adj_to_batched_graphs(dset['input_ids'])
    """

    #device = trainset["input_ids"].device
    device = dset["input_ids"].device
    if use_dgl:
        batched_g = trainer._from_adj_to_batched_graphs(dset['input_ids']).to(device)
    else:
        batched_g = trainer._from_adj_to_batched_sparsify(dset['input_ids']).to(device)
    #trainset["g"] = batched_g    
    dset["g"] = batched_g

    ## 0.3 DiffuserEmbeddings ------------------------------------------------------------
    print("0.3 DiffuserEmbeddings \n")

    # from diffuser.py

    """
    class DiffuserModel(DiffuserPreTrainedModel):
        def __init__(self, config, add_pooling_layer=True, **kwargs):
            super().__init__(config)
            self.config = config        

            if isinstance(config.attention_window, int):
                assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
                assert config.attention_window > 0, "`config.attention_window` has to be positive"
                config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
            else:
                assert len(config.attention_window) == config.num_hidden_layers, (
                    "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                    f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
                )

            self.embeddings = DiffuserEmbeddings(config)
            self.encoder = DiffuserEncoder(config, **kwargs)
            self.pooler = DiffuserPooler(config) if add_pooling_layer else None

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            g=None,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
    """

    # forward pass variable
    #inputs_ids = trainset['input_ids']
    #attention_mask = trainset['attention_mask']
    input_ids = dset['input_ids']
    attention_mask = dset['attention_mask']
    global_attention_mask=None
    head_mask=None
    token_type_ids=None
    position_ids=None
    inputs_embeds=None
    output_attentions=None
    output_hidden_states=None
    return_dict=None

    # self: model.diffuser
    output_attentions = output_attentions if output_attentions is not None else model.diffuser.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.diffuser.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model.diffuser.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # merge `global_attention_mask` and `attention_mask`
    if global_attention_mask is not None:
        attention_mask = model.diffuser._merge_to_attention_mask(attention_mask, global_attention_mask)

    padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = model.diffuser._pad_to_window_size(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        pad_token_id=model.diffuser.config.pad_token_id,
    )

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = model.diffuser.get_extended_attention_mask(attention_mask, input_shape, device)[
        :, 0, 0, :
    ]

    # this is exactly the hidden_state for the forward pass below
    embedding_output = model.diffuser.embeddings(
        input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    )


    """
    encoder_outputs = self.encoder(
        embedding_output,
        g= g,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        padding_len=padding_len,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    """

    ## 0.4 DiffuserEncoder ------------------------------------------------------------
    print("0.4 DiffuserEncoder \n")


    # from diffuser.py
    """
    class DiffuserEncoder(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            self.layer = nn.ModuleList([DiffuserLayer(config, layer_id=i,**kwargs) for i in range(config.num_hidden_layers)])
            self.gradient_checkpointing = False

        def forward(
            self,
            hidden_states,
            g= None,
            attention_mask=None,
            head_mask=None,
            padding_len=0,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ):
    """

    # forward pass variable
    hidden_states = embedding_output
    g = dset['g']
    # attention_mask=None # defined above
    head_mask=None
    padding_len=0
    output_attentions=False
    output_hidden_states=False
    return_dict=True

    is_index_masked = attention_mask < 0
    is_index_global_attn = attention_mask > 0
    is_global_attn = is_index_global_attn.flatten().any().item()

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None  # All local attentions.
    all_global_attentions = () if (output_attentions and is_global_attn) else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(model.diffuser.encoder.layer)
        ), f"The head_mask should be specified for {len(model.diffuser.encoder.layer)} layers, but it is for {head_mask.size()[0]}."

    """
    for idx, layer_module in enumerate(model.diffuser.encoder.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if model.diffuser.encoder.gradient_checkpointing and model.diffuser.encoder.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, is_global_attn, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                head_mask[idx] if head_mask is not None else None,
                is_index_masked,
                is_index_global_attn,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                g=g,
                attention_mask=attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )
        hidden_states = layer_outputs[0]
    """

    # do one layer
    idx = 0
    layer_module = model.diffuser.encoder.layer[idx]     # DiffuserLayer

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if model.diffuser.encoder.gradient_checkpointing and model.diffuser.encoder.training:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, is_global_attn, output_attentions)

            return custom_forward

        layer_outputs = torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            head_mask[idx] if head_mask is not None else None,
            is_index_masked,
            is_index_global_attn,
        )
    else:
        """
        layer_outputs = layer_module(
            hidden_states,
            g=g,
            attention_mask=attention_mask,
            layer_head_mask=head_mask[idx] if head_mask is not None else None,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        """

        # hidden_states # defined above
        # g # defined above
        # attention_mask # defined above
        layer_head_mask=head_mask[idx] if head_mask is not None else None
        # is_index_masked=is_index_masked # defined above
        # is_index_global_attn # defined above
        # is_global_attn # defined above
        # output_attentions # defined above

    # 0.5 DiffuserLayer ------------------------------------------------------------
        print("0.5 DiffuserLayer \n")

        """
        class DiffuserLayer(nn.Module):
            def __init__(self, config, layer_id=0, **kwargs):
                super().__init__()
                self.attention = DiffuserAttention(config,  layer_id, **kwargs)
                self.intermediate = DiffuserIntermediate(config)
                self.output = DiffuserOutput(config)
                self.chunk_size_feed_forward = config.chunk_size_feed_forward
                self.seq_len_dim = 1

            def forward(
                self,
                hidden_states,
                g=None,
                attention_mask=None,
                layer_head_mask=None,
                is_index_masked=None,
                is_index_global_attn=None,
                is_global_attn=None,
                output_attentions=False,
            ):
        """
    
    # 0.6 DiffuserAttention ------------------------------------------------------------
        print("0.6 DiffuserAttention \n")

        # from diffuser_attn.py
        """
        class DiffuserAttention(nn.Module):
            def __init__(self, config, layer_id=0, **kwargs):
                super().__init__()
                assert isinstance(kwargs.get('with_frac'), bool), "with_frac must be boolean"
                self.with_frac = kwargs.get('with_frac')
                if not self.with_frac:
                    self.self = DiffuserSelfAttention(config, layer_id)
                else:
                    assert 0 < kwargs.get('gamma') < 1, "gamma for DiffuserFracSelfAttention is ill-defined!"         
                    gamma = kwargs.get('gamma')
                    self.self = DiffuserFracSelfAttention(config, layer_id, gamma)
                self.output = DiffuserSelfOutput(config)

            def forward(
                self,
                hidden_states,
                g=None,
                attention_mask=None,
                layer_head_mask=None,
                is_index_masked=None,
                is_index_global_attn=None,
                is_global_attn=None,
                output_attentions=False,
            ):
        """

    # 0.7 DiffuserSelfAttention ------------------------------------------------------------
        print("0.7 DiffuserSelfAttention \n")

        # from diffuser_attn.py
        """
        class DiffuserSelfAttention(nn.Module):
            def __init__(self, config, layer_id):
                super().__init__()
                if config.hidden_size % config.num_attention_heads != 0:
                    raise ValueError(
                        f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                        f"heads ({config.num_attention_heads})"
                    )
                self.num_heads = config.num_attention_heads
                self.head_dim = int(config.hidden_size / config.num_attention_heads)
                self.embed_dim = config.hidden_size

                self.query = nn.Linear(config.hidden_size, self.embed_dim)
                self.key = nn.Linear(config.hidden_size, self.embed_dim)
                self.value = nn.Linear(config.hidden_size, self.embed_dim)

                self.dropout = config.attention_probs_dropout_prob

                self.layer_id = layer_id
                attention_window = config.attention_window[self.layer_id]
                assert (
                    attention_window % 2 == 0
                ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
                assert (
                    attention_window > 0
                ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

            def forward(
                self,
                hidden_states,
                g=None,
                attention_mask=None,
                layer_head_mask=None,
                is_index_masked=None,
                is_index_global_attn=None,
                is_global_attn=None,
                output_attentions=False,
            ):
            """
        from models.utils import mask_attention_score

        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)

        # attention_mask (B,N)
        # project hidden states
        query_vectors = layer_module.attention.self.query(hidden_states)
        key_vectors = layer_module.attention.self.key(hidden_states)
        value_vectors = layer_module.attention.self.value(hidden_states)   # (N,B,HD)      

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == layer_module.attention.self.embed_dim
        ), f"hidden_states should have embed_dim = {layer_module.attention.self.embed_dim}, but has {embed_dim}"     

        dims_all = [seq_len, batch_size, layer_module.attention.self.num_heads, layer_module.attention.self.head_dim,
                    embed_dim]   
        dropout_args = [layer_module.attention.self.dropout, layer_module.attention.self.training]  

    return query_vectors, key_vectors, value_vectors, attention_mask, g, dims_all, dropout_args