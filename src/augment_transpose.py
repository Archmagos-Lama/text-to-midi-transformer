import os
import random
from pathlib import Path
from miditoolkit import MidiFile
from tqdm import tqdm


INPUT_DIR  = Path(r"data/touhou_midi_collection")
TRAIN_DIR  = Path(r"data/touhou_augmented_train")
VAL_DIR    = Path(r"data/touhou_augmented_val")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

TRANSPOSE_RANGE = range(-6, 7)
VAL_RATIO = 0.2
SEED = 42


midi_files = []
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith((".mid", ".midi")):
            midi_files.append(Path(root) / f)

print("Found MIDI:", len(midi_files))

random.seed(SEED)
random.shuffle(midi_files)
n_val = max(1, int(len(midi_files) * VAL_RATIO))
val_files  = set(midi_files[:n_val])
train_files = midi_files[n_val:]

print(f"Train: {len(train_files)}  Val: {len(val_files)}")


def copy_midi(midi_path, out_dir, relative_to):
    rel = midi_path.relative_to(relative_to)
    out = out_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    MidiFile(midi_path).dump(out)


def augment_midi(midi_path, out_dir, relative_to):
    rel = midi_path.relative_to(relative_to)
    for shift in TRANSPOSE_RANGE:
        midi_copy = MidiFile(midi_path)
        valid = True
        for inst in midi_copy.instruments:
            for note in inst.notes:
                new_pitch = note.pitch + shift
                if new_pitch < 0 or new_pitch > 127:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        out = out_dir / rel.parent / (midi_path.stem + f"_tr{shift:+d}.mid")
        out.parent.mkdir(parents=True, exist_ok=True)
        midi_copy.dump(out)


print("Augmenting train set...")
for p in tqdm(train_files):
    try:
        augment_midi(p, TRAIN_DIR, INPUT_DIR)
    except Exception:
        print("skip:", p)

print("Copying val set (no augmentation)...")
for p in tqdm(val_files):
    try:
        copy_midi(p, VAL_DIR, INPUT_DIR)
    except Exception:
        print("skip:", p)

print("Done")
print("Train files in output:", sum(1 for _ in TRAIN_DIR.rglob("*.mid")))
print("Val files in output:  ", sum(1 for _ in VAL_DIR.rglob("*.mid")))
