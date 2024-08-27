import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


# Load the data
df = pd.read_csv("emobank.csv")
data = df.sample(frac=1, random_state=42).reset_index(drop=True)
all_texts = data["text"].tolist()
valence_labels = data["V"].tolist()
arousal_labels = data["A"].tolist()
all_labels = [[x, y] for x, y in zip(valence_labels, arousal_labels)]
print(f"Total number of sentences: {len(all_texts)}")
print(f"Total number of labels: {len(all_labels)}")
print(f"Shape of labels: {np.array(all_labels).shape}")

print(f"example of texts: {all_texts[4]}")
print(f"example of labels: {all_labels[4]}")

train_texts = all_texts[: int(0.8 * len(all_texts))]
train_labels = all_labels[: int(0.8 * len(all_texts))]

val_texts = all_texts[int(0.8 * len(all_texts)) : int(0.9 * len(all_texts))]
val_labels = all_labels[int(0.8 * len(all_labels)) : int(0.9 * len(all_labels))]

test_texts = all_texts[int(0.9 * len(all_texts)) :]
test_labels = all_labels[int(0.9 * len(all_labels)) :]


# Assuming you have a dataset class similar to the one described before
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label,
        }


# Set hyperparameters
batch_size = 16
max_length = 128
learning_rate = 2e-5
num_epochs = 10

# Load the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large", num_labels=2
)  # Regression: num_labels=2

model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)


# Create datasets and data loaders
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_model_path = "best_roberta_large.pth"
best_val_loss = 100

patience = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # to make sure it's using the gpu
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)  # Assuming the output is logits
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += criterion(outputs.logits, labels).item()

    val_loss /= len(val_dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}: Validation Accuracy: {val_loss}")

    # Save the best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), best_model_path)
        patience = 0
    else:
        patience += 1
        if patience > 5:
            break

    # print the best val loss
    print(f"Best val loss: {best_val_loss}")
