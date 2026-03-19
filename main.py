"""
main.py
Two-phase training with evaluation metrics
Updated for Dual Encoder: BERT + MuRIL
"""

import json
import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from model import DepressionModel
from utils import CLASS_NAMES, preprocess_text, severity_score


# ───────────────── CONFIG ─────────────────

DATA_PATH      = "data_to_be_cleansed.csv"
CACHE_PATH     = "bert_cache.pt"

PHASE1_EPOCHS  = 8
PHASE2_EPOCHS  = 10

HEAD_LR        = 2e-4
BERT_LR        = 2e-5

POSTS_PER_USER = 5
EMBED_BATCH    = 64


# ───────────────── 1. LOAD DATA ─────────────────

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["text"]  = df["text"].fillna("")
df["title"] = df["title"].fillna("")

df["combined"] = (
        df["title"].str.strip() + " " +
        df["text"].str.strip()
).apply(preprocess_text)

df["label"] = df["target"].clip(upper=3).astype(int)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())


# ───────────────── 2. BUILD SESSIONS ─────────────────

df["uid"] = df.index // POSTS_PER_USER

sessions = []

for _, g in df.groupby("uid"):

    sessions.append({
        "texts": g["combined"].tolist(),
        "label": int(g["label"].max())
    })

print("\nTotal sessions:", len(sessions))


# ───────────────── 3. MODEL ─────────────────

print("\nInitialising model...")

model = DepressionModel()


# ───────────────── 4. CACHE EMBEDDINGS ─────────────────
# FIX 1: Changed model.encoder.bert.eval() 
# to model.encoder.eval() because encoder is now
# DualEncoder (BERT + MuRIL), not a single BERT model

if os.path.exists(CACHE_PATH):

    print("\nLoading cached embeddings...")

    cache = torch.load(CACHE_PATH, map_location="cpu")

    all_embeddings = cache["embeddings"]

else:

    print("\nComputing BERT + MuRIL embeddings...")
    print("(This will take longer as both BERT and MuRIL run on each batch)")

    all_texts = []

    for s in sessions:
        all_texts.extend(s["texts"])

    all_embeddings = []

    # FIX 1: use model.encoder.eval() instead of model.encoder.bert.eval()
    model.encoder.eval()

    with torch.no_grad():

        for start in tqdm(range(0, len(all_texts), EMBED_BATCH)):

            batch = all_texts[start:start+EMBED_BATCH]

            emb = model.encoder(batch)   # returns 1536-dim (768 BERT + 768 MuRIL)

            all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    torch.save({"embeddings": all_embeddings}, CACHE_PATH)


# rebuild session embeddings

posts_per_session = [len(s["texts"]) for s in sessions]

session_embs = []

ptr = 0

for n in posts_per_session:

    session_embs.append(all_embeddings[ptr:ptr+n])

    ptr += n


# ───────────────── 5. TRAIN / TEST SPLIT ─────────────────

indices = list(range(len(sessions)))

labels_split = [sessions[i]["label"] for i in indices]

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels_split
)

print("\nTrain:", len(train_idx), " Test:", len(test_idx))


# ───────────────── 6. CLASS WEIGHTS ─────────────────

counts = [0]*4

for i in train_idx:
    counts[sessions[i]["label"]] += 1

total = sum(counts)

weights = torch.tensor(
    [min(total/(4*c),5.0) if c>0 else 1.0 for c in counts],
    dtype=torch.float32
).to(model.device)

print("Class weights:", weights)

loss_cls = torch.nn.CrossEntropyLoss(weight=weights)

loss_reg = torch.nn.MSELoss()


# ───────────────── TRAINING FUNCTION ─────────────────

def run_epoch(use_cache):

    total_loss = 0
    correct    = 0

    for i in train_idx:

        if use_cache:
            emb = session_embs[i].unsqueeze(0).to(model.device)
        else:
            emb = model.embed(sessions[i]["texts"]).unsqueeze(0).to(model.device)

        cls_target = torch.tensor(
            [sessions[i]["label"]],
            dtype=torch.long
        ).to(model.device)

        reg_target = torch.tensor(
            [[severity_score(sessions[i]["label"])]],
            dtype=torch.float32
        ).to(model.device)

        optimizer.zero_grad()

        logits, score = model.forward(emb)

        loss = loss_cls(logits, cls_target) + 0.7 * loss_reg(score, reg_target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        correct += int(logits.argmax(dim=-1).item() == sessions[i]["label"])

    acc = correct / len(train_idx) * 100

    return total_loss / len(train_idx), acc


# ───────────────── PHASE 1 ─────────────────

print("\n=== PHASE 1 : Training heads (BERT + MuRIL frozen) ===")

optimizer = torch.optim.AdamW(
    model.head_parameter_groups(HEAD_LR),
    weight_decay=1e-2
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=[HEAD_LR]*len(optimizer.param_groups),
    total_steps=len(train_idx)*PHASE1_EPOCHS,
    pct_start=0.1
)

for epoch in range(PHASE1_EPOCHS):

    model.train()

    # FIX 2: use model.encoder.eval() instead of model.encoder.bert.eval()
    # because encoder is now DualEncoder containing both BERT and MuRIL
    model.encoder.eval()

    loss, acc = run_epoch(True)

    print(f"[P1] Epoch {epoch+1}/{PHASE1_EPOCHS} loss={loss:.4f} acc={acc:.2f}%")


# ───────────────── PHASE 2 ─────────────────

print("\n=== PHASE 2 : Fine-tuning top layers of BERT + MuRIL ===")

model.encoder.unfreeze_top()

optimizer = torch.optim.AdamW(
    model.finetune_parameter_groups(BERT_LR, HEAD_LR),
    weight_decay=1e-2
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=[BERT_LR] + [HEAD_LR]*(len(optimizer.param_groups)-1),
    total_steps=len(train_idx)*PHASE2_EPOCHS,
    pct_start=0.2
)

for epoch in range(PHASE2_EPOCHS):

    model.train()
    model.encoder.train()

    loss, acc = run_epoch(False)

    print(f"[P2] Epoch {epoch+1}/{PHASE2_EPOCHS} loss={loss:.4f} acc={acc:.2f}%")


model.save()

print("\nModel saved: depression_model.pt")


# ───────────────── EVALUATION ─────────────────

print("\nEvaluating model...")

model.eval()
model.encoder.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for i in tqdm(test_idx):

        emb = model.embed(sessions[i]["texts"]).unsqueeze(0).to(model.device)

        logits, _ = model.forward(emb)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred = int(np.argmax(probs))

        y_pred.append(pred)
        y_true.append(sessions[i]["label"])


report = classification_report(y_true, y_pred, output_dict=True)

cm = confusion_matrix(y_true, y_pred)

with open("metrics.json", "w") as f:

    json.dump(
        {
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        },
        f,
        indent=2
    )

print("\nClassification Report:\n")

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("\nConfusion Matrix:\n")

print(cm)

print("\nMetrics saved to metrics.json")