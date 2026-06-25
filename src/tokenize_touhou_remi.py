import os
from pathlib import Path
import torch
from miditoolkit import MidiFile
from miditok import REMI, TokenizerConfig
from tqdm import tqdm


TRAIN_DIR = Path(r"data/touhou_augmented_train")
VAL_DIR   = Path(r"data/touhou_augmented_val")
OUT_DIR   = Path("dataset_remi")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PT       = OUT_DIR / "train_tokens.pt"
VAL_PT         = OUT_DIR / "val_tokens.pt"
TOKENIZER_JSON = OUT_DIR / "tokenizer.json"

config = TokenizerConfig(
    use_chords=True,
    use_tempos=True,
    use_programs=True,
    use_time_signatures=False,
    use_sustain_pedals=False,
    use_pitch_bends=False,
    beat_res={(0, 4): 8, (4, 12): 4},
    nb_velocities=32,
    chord_tokens_with_root_note=True,
)

tokenizer = REMI(config)
tokenizer.save_params(TOKENIZER_JSON)


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
            all_ids.extend(tok_seq.ids)
        except Exception:
            skipped += 1
            print("skip:", midi_path)
    print(f"{label}: {len(all_ids)} tokens, {skipped} skipped")
    return torch.tensor(all_ids, dtype=torch.long)


train_files = collect_midis(TRAIN_DIR)
val_files   = collect_midis(VAL_DIR)
print(f"Train MIDI: {len(train_files)}  Val MIDI: {len(val_files)}")

train_tokens = tokenize_files(train_files, "train")
val_tokens   = tokenize_files(val_files,   "val")

torch.save(train_tokens, TRAIN_PT)
torch.save(val_tokens,   VAL_PT)

print("\nDone")
print("Train tokens:", len(train_tokens), "->", TRAIN_PT)
print("Val tokens:  ", len(val_tokens),   "->", VAL_PT)
print("Vocab size:  ", tokenizer.vocab_size)
