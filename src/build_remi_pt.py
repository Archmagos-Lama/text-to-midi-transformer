import torch
import json
from pathlib import Path
from tqdm import tqdm

JSON_DIR = "remi_json"
OUT_PT = "train_corpus_remi.pt"

all_ids = []

json_files = list(Path(JSON_DIR).rglob("*.json"))

for jf in tqdm(json_files):
    data = json.load(open(jf))
    all_ids.extend(data["ids"])

train_tensor = torch.tensor(all_ids, dtype=torch.long)
torch.save(train_tensor, OUT_PT)

print("Saved:", OUT_PT)
print("Total tokens:", train_tensor.shape[0])