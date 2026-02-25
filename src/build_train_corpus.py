import os
import json
import torch
from tqdm import tqdm

# ===================== CONFIG =====================

EVENT_ROOT = "data/events"
VOCAB_PATH = "data/vocab/vocab.json"

OUT_DIR = "data/pt"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.99  # 99% train, 1% val

# ==================================================

# ---------- load vocab ----------
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

# ---------- collect event files ----------
def iter_event_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".txt"):
                yield os.path.join(dirpath, fn)

all_event_files = list(iter_event_files(EVENT_ROOT))
print("Total event files:", len(all_event_files))

# ---------- build token stream ----------
all_tokens = []

for path in tqdm(all_event_files, desc="Building token stream"):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # 可选：加入 <START> / <END>
    lines = ["<START>"] + lines + ["<END>"]

    for token in lines:
        if token in token_to_id:
            all_tokens.append(token_to_id[token])
        else:
            # 忽略未知 token（安全）
            continue

all_tokens = torch.tensor(all_tokens, dtype=torch.long)

print("Total tokens:", len(all_tokens))
print("Vocab size :", len(vocab))

# ---------- split train / val ----------
split_idx = int(len(all_tokens) * TRAIN_RATIO)

train_data = all_tokens[:split_idx]
val_data   = all_tokens[split_idx:]

print("Train tokens:", len(train_data))
print("Val tokens  :", len(val_data))

# ---------- save ----------
torch.save(train_data, os.path.join(OUT_DIR, "train.pt"))
torch.save(val_data,   os.path.join(OUT_DIR, "val.pt"))

print("Saved:")
print(" ", os.path.join(OUT_DIR, "train.pt"))
print(" ", os.path.join(OUT_DIR, "val.pt"))