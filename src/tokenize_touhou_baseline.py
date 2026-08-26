"""
Baseline tokenization: same vocab=741, same BOS/EOS, but Bar_None is kept as-is.
Single controlled variable vs tokenize_touhou_remi.py: no BARPOS_k injection.
Outputs go to dataset_remi_baseline/.
"""
import json
import os
from pathlib import Path

import torch
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from tqdm import tqdm

TRAIN_DIR = Path(r"data/touhou_augmented_train")
VAL_DIR   = Path(r"data/touhou_augmented_val")
OUT_DIR   = Path("dataset_remi_baseline")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PT       = OUT_DIR / "train_tokens.pt"
VAL_PT         = OUT_DIR / "val_tokens.pt"
TOKENIZER_JSON = OUT_DIR / "tokenizer.json"
META_JSON      = OUT_DIR / "meta.json"

MAX_BARS = 128

# Identical config to BARPOS version — same vocab size 741
config = TokenizerConfig(
    use_chords=True,
    use_tempos=True,
    use_programs=True,
    use_time_signatures=False,
    use_sustain_pedals=False,
    use_pitch_bends=False,
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    chord_tokens_with_root_note=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"] + [f"BARPOS_{i}" for i in range(MAX_BARS)],
)

tokenizer = REMI(config)
tokenizer.save(TOKENIZER_JSON)

bar_pos_start = tokenizer["BARPOS_0"]
bos_id        = tokenizer["BOS_None"]
eos_id        = tokenizer["EOS_None"]

print(f"Vocab size: {tokenizer.vocab_size}  (BARPOS tokens present but NOT injected)")


def collect_midis(directory):
    files = []
    for root, dirs, fnames in os.walk(directory):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(Path(root) / f)
    return files


def tokenize_files(midi_files, label):
    all_ids = []
    skipped = 0
    for midi_path in tqdm(midi_files, desc=label):
        try:
            midi = MidiFile(midi_path)
            tok_seq = tokenizer(midi)
            if not tok_seq.ids:
                skipped += 1
                continue
            # Keep Bar_None as-is — no injection
            ids = [bos_id] + tok_seq.ids + [eos_id]
            all_ids.extend(ids)
        except Exception:
            skipped += 1
            print("skip:", midi_path)
    print(f"{label}: {len(all_ids)} tokens, {skipped} skipped")
    return torch.tensor(all_ids, dtype=torch.long)


train_files = collect_midis(TRAIN_DIR)
val_files   = collect_midis(VAL_DIR)
print(f"Train MIDI: {len(train_files)}  Val MIDI: {len(val_files)}")

train_tokens = tokenize_files(train_files, "train")
val_tokens   = tokenize_files(val_files, "val")

torch.save(train_tokens, TRAIN_PT)
torch.save(val_tokens, VAL_PT)

meta = {
    "vocab_size": tokenizer.vocab_size,
    "bar_pos_start": bar_pos_start,
    "max_bars": MAX_BARS,
    "barpos_injected": False,
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print("\nDone")
print("Train tokens:", len(train_tokens), "->", TRAIN_PT)
print("Val tokens:  ", len(val_tokens), "->", VAL_PT)
print("Vocab size:  ", tokenizer.vocab_size)
