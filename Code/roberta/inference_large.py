import os
import torch
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import logging

# Set the logging level to ERROR to avoid unnecessary outputs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set the environment variable to use only GPU 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def emotion_prediction(text_to_predict):
    # Load the trained model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large", num_labels=2
    )

    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Download the fine-tuned weights from Hugging Face (cached after first use)
    model_path = hf_hub_download(
        repo_id="RuiSumida/LUFY", filename="best_roberta_large.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    max_length = 128

    # Tokenize the input text
    encoding = tokenizer(
        text_to_predict,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Perform a forward pass and get the prediction
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = outputs.logits.squeeze()

    return prediction
