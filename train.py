import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

########################################
# DEVICE
########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

########################################
# LOAD DATASET
########################################

df = pd.read_csv("data_to_be_cleansed.csv")

# combine title and text
texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()

# labels 0-4
labels = df["target"].astype(int).tolist()

########################################
# TRAIN TEST SPLIT
########################################

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.1,
    random_state=42
)

########################################
# TOKENIZER
########################################

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

########################################
# DATASET CLASS
########################################

class DepressionDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

########################################
# DATA LOADERS
########################################

train_dataset = DepressionDataset(train_texts, train_labels)
val_dataset = DepressionDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

########################################
# MODEL
########################################

class BertBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        # 5 classes
        self.fc = nn.Linear(256, 5)

    def forward(self, input_ids, attention_mask):

        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.dropout(x)

        x = self.fc(x)

        return x

########################################
# INITIALIZE MODEL
########################################

model = BertBiLSTM().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

########################################
# TRAINING LOOP
########################################

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

########################################
# VALIDATION
########################################

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("Validation Accuracy:", accuracy)

########################################
# SAVE MODEL
########################################

torch.save(model.state_dict(), "depression_model.pt")

print("Model saved successfully as depression_model.pt")
