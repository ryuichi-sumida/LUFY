import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel
import numpy as np

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#calculating perplexity for the second sentence
def calculate_perplexity(input_sentence1, input_sentence2):
    # Load the GPT-2 tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    input_sentence = input_sentence1 + ' ' + input_sentence2

    # Tokenize the input sentence
    input_ids1 = tokenizer(input_sentence1, return_tensors="pt", clean_up_tokenization_spaces=True)["input_ids"].to(device)
    input_ids2 = tokenizer(input_sentence2, return_tensors="pt", clean_up_tokenization_spaces=True)["input_ids"].to(device)
    input_ids = tokenizer(input_sentence, return_tensors="pt", clean_up_tokenization_spaces=True)["input_ids"].to(device)


    tensor_length1 = input_ids1.shape[1]
    tensor_length2 = input_ids2.shape[1]
    tensor_length = input_ids.shape[1]

    # Generate logits from the model
    with torch.no_grad():
        logits = model(input_ids).logits

    # Apply softmax to calculate probabilities
    probabilities = F.softmax(logits, dim=-1)

    list_probabilities = []

    #creating a list of probabilities for the second sentence
    for i in range(tensor_length1-1, tensor_length-1):
        #get the probability for the i+1th token

        token_id = input_ids[0][i+1]
        numbers = probabilities[0][i]

        probability = numbers[token_id]

        probability = torch.round(probability, decimals=10)

        list_probabilities.append(probability.cpu().numpy()) # Move to CPU for NumPy compatibility

    list_probabilities = np.array(list_probabilities)
    num_words = len(list_probabilities)

    #to avoid log(0) error
    filtered_probabilities = [p for p in list_probabilities if p > 0]
    num_words = len(filtered_probabilities)
    
    # Calculate the sum of log probabilities using NumPy
    log_sum = np.sum(np.log2(filtered_probabilities))
    
    # Calculate perplexity
    perplexity = 2 ** (- (1 / num_words) * log_sum)
    
    return perplexity

