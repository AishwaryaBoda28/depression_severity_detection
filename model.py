"""
model.py
────────
Architecture: BERT + MuRIL dual encoder → BiLSTM → dual-head (cls + reg)

Uses BOTH:
- bert-base-uncased       : Strong general English understanding
- google/muril-base-cased : Strong Indian + multilingual understanding

Their outputs are concatenated → 1536-dim → BiLSTM
This gives better coverage than either model alone.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

CHECKPOINT      = "depression_model.pt"
BERT_ENCODER    = "bert-base-uncased"
MURIL_ENCODER   = "google/muril-base-cased"
UNFREEZE_LAST_N = 2


############################################
# SINGLE ENCODER (reusable for both)
############################################

class SingleEncoder(nn.Module):

    def __init__(self, model_name, device):
        super().__init__()
        self.device = device

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.dim       = self.model.config.hidden_size  # 768 for both
        self.freeze_all()

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_top(self):
        total = len(self.model.encoder.layer)
        for i in range(total - UNFREEZE_LAST_N, total):
            for p in self.model.encoder.layer[i].parameters():
                p.requires_grad = True
        if self.model.pooler is not None:
            for p in self.model.pooler.parameters():
                p.requires_grad = True

    def forward(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        out    = self.model(**tokens).last_hidden_state
        mask   = tokens["attention_mask"].unsqueeze(-1).float()

        # Safe mean pooling
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return pooled   # [batch, 768]


############################################
# DUAL ENCODER (BERT + MuRIL combined)
############################################

class DualEncoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

        self.bert_encoder  = SingleEncoder(BERT_ENCODER,  device)
        self.muril_encoder = SingleEncoder(MURIL_ENCODER, device)

        # Combined dim = 768 + 768 = 1536
        self.dim = self.bert_encoder.dim + self.muril_encoder.dim

    def freeze_all(self):
        self.bert_encoder.freeze_all()
        self.muril_encoder.freeze_all()

    def unfreeze_top(self):
        self.bert_encoder.unfreeze_top()
        self.muril_encoder.unfreeze_top()

    def forward(self, texts):
        bert_out  = self.bert_encoder(texts)   # [batch, 768]
        muril_out = self.muril_encoder(texts)  # [batch, 768]

        # Concatenate → [batch, 1536]
        combined  = torch.cat([bert_out, muril_out], dim=-1)

        return combined


############################################
# DEPRESSION MODEL
############################################

class DepressionModel(nn.Module):

    NUM_CLASSES = 4

    def __init__(self):
        super().__init__()

        self.device  = "cuda" if torch.cuda.is_available() else "cpu"

        # Dual encoder: BERT + MuRIL
        self.encoder = DualEncoder(self.device)
        self.encoder.to(self.device)

        emb_dim = self.encoder.dim  # 1536

        # BiLSTM input is now 1536 instead of 768
        self.lstm = nn.LSTM(
            emb_dim,
            128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.cls_head = nn.Linear(128, self.NUM_CLASSES)

        self.reg_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.to(self.device)

    def parameters(self, recurse=True):
        return [p for p in super().parameters(recurse=recurse) if p.requires_grad]

    def forward(self, x):
        x = x.to(self.device)

        lstm_out, _ = self.lstm(x)
        pooled      = lstm_out.mean(dim=1)
        features    = self.fc(pooled)
        logits      = self.cls_head(features)
        score       = self.reg_head(features) * 100.0

        return logits, score

    def embed(self, texts):
        """Get combined BERT + MuRIL embedding"""
        return self.encoder(texts)

    def predict(self, texts):
        self.eval()
        with torch.no_grad():
            emb           = self.embed(texts).unsqueeze(0).to(self.device)
            logits, score = self.forward(emb)
            probs         = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()[0], float(score.cpu().numpy()[0][0])

    def predict_proba(self, texts):
        outputs = []
        for t in texts:
            probs, _ = self.predict([t] * 5)
            outputs.append(probs)
        return np.array(outputs)

    def save(self, path=CHECKPOINT):
        torch.save(self.state_dict(), path)

    def load(self, path=CHECKPOINT):
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, strict=False)
        self.eval()

    # ── Phase 1: train heads only ──
    def head_parameter_groups(self, lr):
        return [
            {"params": self.lstm.parameters(),     "lr": lr},
            {"params": self.fc.parameters(),       "lr": lr},
            {"params": self.cls_head.parameters(), "lr": lr},
            {"params": self.reg_head.parameters(), "lr": lr},
        ]

    # ── Phase 2: fine-tune top layers of BOTH encoders ──
    def finetune_parameter_groups(self, bert_lr, head_lr):
        self.encoder.unfreeze_top()
        return [
            {"params": self.encoder.bert_encoder.model.parameters(),  "lr": bert_lr},
            {"params": self.encoder.muril_encoder.model.parameters(), "lr": bert_lr},
            {"params": self.lstm.parameters(),                        "lr": head_lr},
            {"params": self.fc.parameters(),                          "lr": head_lr},
            {"params": self.cls_head.parameters(),                    "lr": head_lr},
            {"params": self.reg_head.parameters(),                    "lr": head_lr},
        ]