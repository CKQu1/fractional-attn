import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu

# Download NLTK resources
nltk.download('punkt')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Example input sentences
inputs = ["Hello world!", "How are you?"]

# Tokenize inputs
encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

# Generate translations
with torch.no_grad():
    outputs = model.generate(**encoded_inputs)
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Tokenize translations
translations = [nltk.word_tokenize(t) for t in translations]

# Example reference translations
references = [["Hallo Welt!", "Wie geht es Ihnen?"]]

# Tokenize references
references = [[nltk.word_tokenize(ref) for ref in ref_list] for ref_list in references][0]

# Compute BLEU score
bleu_score = corpus_bleu(references, translations)
print("BLEU score:", bleu_score)