import os
import json
from tqdm import tqdm

EVENT_ROOT = "data/events"
VOCAB_PATH = "data/vocab/vocab.json"
OUT_ROOT   = "data/tokens"

os.makedirs(OUT_ROOT, exist_ok=True)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

def iter_event_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".txt"):
                yield os.path.join(dirpath, fn)

def rel_out_path(in_path, in_root, out_root, ext):
    rel = os.path.relpath(in_path, in_root)
    rel = os.path.splitext(rel)[0] + ext
    out = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out

event_files = list(iter_event_files(EVENT_ROOT))

for evt_path in tqdm(event_files, desc="Tokenizing events"):
    with open(evt_path, "r", encoding="utf-8") as f:
        events = [l.strip() for l in f if l.strip()]

    tokens = []
    for e in events:
        if e not in vocab:
            raise RuntimeError(f"OOV token: {e} in {evt_path}")
        tokens.append(vocab[e])

    out_path = rel_out_path(evt_path, EVENT_ROOT, OUT_ROOT, ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f)
