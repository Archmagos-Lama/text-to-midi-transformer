import os
import json
import torch
from tqdm import tqdm

TOKEN_ROOT = "data/tokens"
OUT_PATH   = "data/train_corpus.pt"

all_tokens = []

def iter_token_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                yield os.path.join(dirpath, fn)

token_files = list(iter_token_files(TOKEN_ROOT))

for path in tqdm(token_files, desc="Building corpus"):
    with open(path, "r", encoding="utf-8") as f:
        seq = json.load(f)

    all_tokens.extend(seq)

tensor = torch.tensor(all_tokens, dtype=torch.long)
torch.save(tensor, OUT_PATH)

print("Total tokens:", tensor.numel())
print("Saved:", OUT_PATH)
