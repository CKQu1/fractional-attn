from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer_src = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')  # German tokenizer
tokenizer_trg = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')  # English tokenizer

print(tokenizer_src.vocab_size)
print(tokenizer_src.pad_token_id)

# trainloader, testloader = prepare_data(tokenizer_src, tokenizer_trg, batch_size=batch_size, num_workers=ddp_world_size, max_length=config["max_length"]) 