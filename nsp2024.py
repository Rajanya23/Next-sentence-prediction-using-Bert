from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# Example sentences
sentence_1 = "i love"
sentence_2 = "i love"

# Tokenize input
inputs = tokenizer(sentence_1, sentence_2, return_tensors='pt')

# Perform NSP
outputs = model(**inputs)

# Get the predicted probability that sentence_2 follows sentence_1
probability = torch.softmax(outputs.logits, dim=1)[0, 0].item()

print(f"Probability that sentence_2 follows sentence_1: {probability:.2f}")
