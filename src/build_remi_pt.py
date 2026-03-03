import json
import torch
from pathlib import Path
from miditok import REMI
from miditok.classes import TokenizerConfig

JSON_DIR = "remi_json"
OUT_PATH = "train_corpus_remi.pt"

tokenizer = REMI(params="data/tokenizer_config.json")
BOS_ID = tokenizer.vocab["BOS_None"]
EOS_ID = tokenizer.vocab["EOS_None"]
all_ids = []

for jf in Path(JSON_DIR).glob("*.json"):
    with open(jf) as f:
        data = json.load(f)

    ids = data["ids"]

    for seq in ids:
        all_ids.append(BOS_ID)
        all_ids.extend(seq)
        all_ids.append(EOS_ID)

print("Total tokens:", len(all_ids))

train_tensor = torch.tensor(all_ids, dtype=torch.long)
torch.save(train_tensor, OUT_PATH)

print("Saved:", OUT_PATH)
print("Final vocab_size =", int(train_tensor.max().item()) + 1)