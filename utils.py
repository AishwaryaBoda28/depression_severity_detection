import re
import random
import numpy as np
import torch
import emoji

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA = re.compile(r"[^a-zA-Z0-9\s!?.,:]")

def preprocess_text(text):

    text = str(text).lower()

    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)

    text = emoji.demojize(text)

    text = NON_ALPHA.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


CLASS_NAMES = [
    "Not Depressed",
    "Mild",
    "Moderate",
    "Severe"
]


def severity_score(label):

    scores = [10.0, 40.0, 70.0, 90.0]

    label = int(label)

    if label < 0 or label >= len(scores):
        label = 3

    return scores[label]


# ───────── EMOJI DETECTION ─────────

def extract_emojis(text):
    """Return list of emojis found in text"""
    return [char for char in text if char in emoji.EMOJI_DATA]
