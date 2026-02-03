import os
import json
from collections import Counter

EVENT_ROOT = "data/events"
OUT_VOCAB  = "data/vocab/vocab.json"

os.makedirs("data/vocab", exist_ok=True)

token_counter = Counter()

def iter_event_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".txt"):
                yield os.path.join(dirpath, fn)

# ===== collect file list =====
event_files = list(iter_event_files(EVENT_ROOT))
total_files = len(event_files)

print(f"[build_vocab] total event files: {total_files}")

# ===== scan events =====
for i, evt_path in enumerate(event_files, 1):
    with open(evt_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                token_counter[tok] += 1

    # 进度心跳（每 100 个文件）
    if i % 100 == 0 or i == total_files:
        print(f"[build_vocab] processed {i}/{total_files}")

# ===== build vocab =====
vocab = {}
idx = 0

# special tokens（如需要）
for tok in ["<PAD>", "<START>", "<END>"]:
    vocab[tok] = idx
    idx += 1

def tok_key(t):
    if t.startswith("TIME_SIG_"):
        return (0, t)
    if t.startswith("INST_"):
        return (1, t)
    if t.startswith("NOTE_ON_"):
        return (2, int(t.split("_")[-1]))
    if t.startswith("DUR_"):
        return (3, float(t.split("_")[-1]))
    return (9, t)

for tok in sorted(token_counter.keys(), key=tok_key):
    vocab[tok] = idx
    idx += 1

with open(OUT_VOCAB, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print("==== vocab built from events ====")
print("Total tokens:", len(vocab))
print("Written:", OUT_VOCAB)
