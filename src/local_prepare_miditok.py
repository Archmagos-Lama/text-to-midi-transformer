import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from random import shuffle

from transformers.trainer_utils import set_seed

from miditok import REMI, TokenizerConfig
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset


BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack
    "num_tempos": 32,
    "tempo_range": (50, 200),
}


def list_midis(midi_root: Path) -> list[Path]:
    midi_root = midi_root.resolve()
    paths = list(midi_root.glob("**/*.mid")) + list(midi_root.glob("**/*.midi"))
    uniq = sorted({p.resolve() for p in paths})
    return uniq


def split_three(paths: list[Path], valid_ratio: float, test_ratio: float, seed: int):
    set_seed(seed)
    paths = deepcopy(paths)
    shuffle(paths)
    n = len(paths)
    n_valid = round(n * valid_ratio)
    n_test = round(n * test_ratio)
    valid = paths[:n_valid]
    test = paths[n_valid:n_valid + n_test]
    train = paths[n_valid + n_test:]
    return train, valid, test


def dump_manifest(out_dir: Path, train, valid, test):
    manifest = {
        "train": [str(p) for p in train],
        "valid": [str(p) for p in valid],
        "test":  [str(p) for p in test],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def split_flat(files_paths, tokenizer, save_dir, max_seq_len, num_overlap_bars):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=save_dir,
        max_seq_len=max_seq_len,
        num_overlap_bars=num_overlap_bars,
    )

    for p in save_dir.rglob("*.mid"):
        flat_path = save_dir / p.name
        if p != flat_path:
            flat_path.parent.mkdir(exist_ok=True)
            p.rename(flat_path)

def main():
    MIDI_ROOT = "data/midi"
    OUT_DIR = "data/miditok_out"

    SEED = 777
    VOCAB_SIZE = 2000

    MAX_SEQ_LEN = 1024
    NUM_OVERLAP_BARS = 2

    VALID_RATIO = 0.15
    TEST_RATIO = 0.15

    AUGMENT = False

    SKIP_TRAIN = False

    TRAIN_CHUNK_SIZE = 2000   
    START_PART = 0         
    ONLY_SUBSET = None       

    midi_root = Path(MIDI_ROOT)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_paths = list(midi_root.glob("**/*.mid")) + list(midi_root.glob("**/*.midi"))
    midi_paths = sorted({p.resolve() for p in midi_paths})
    print(f"Found {len(midi_paths)} MIDI files")

    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)

    if not SKIP_TRAIN:
        print("Training tokenizer...")
        set_seed(SEED)
        tokenizer.train(vocab_size=VOCAB_SIZE, files_paths=midi_paths)
        tokenizer.save_params(out_dir / "tokenizer.json")
    else:
        print("Skipping tokenizer training...")

    set_seed(SEED)
    shuffle(midi_paths)

    n = len(midi_paths)
    n_valid = round(n * VALID_RATIO)
    n_test = round(n * TEST_RATIO)

    midi_valid = midi_paths[:n_valid]
    midi_test = midi_paths[n_valid:n_valid + n_test]
    midi_train = midi_paths[n_valid + n_test:]

    print("train:", len(midi_train))
    print("valid:", len(midi_valid))
    print("test :", len(midi_test))

    chunks_root = out_dir / "data_chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    subsets = []

    if not SKIP_TRAIN:
        subsets.append((midi_train, "train"))

    subsets.append((midi_valid, "valid"))
    subsets.append((midi_test, "test"))

    for files_paths, subset_name in subsets:

        if ONLY_SUBSET and subset_name != ONLY_SUBSET:
            continue

        subset_dir = chunks_root / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing subset: {subset_name}")

        total = len(files_paths)

        for part_idx, start in enumerate(range(0, total, TRAIN_CHUNK_SIZE)):

            if part_idx < START_PART:
                continue

            end = min(start + TRAIN_CHUNK_SIZE, total)
            part = files_paths[start:end]

            print(f"  Part {part_idx} | files {start} ~ {end}")

            split_files_for_training(
                files_paths=part,
                tokenizer=tokenizer,
                save_dir=subset_dir,
                max_seq_len=MAX_SEQ_LEN,
                num_overlap_bars=NUM_OVERLAP_BARS,
            )
            progress_file = subset_dir / f"_part_{part_idx}.done"
            progress_file.write_text("done")

        if AUGMENT:
            print(f"Augmenting {subset_name}...")
            augment_dataset(
                subset_dir,
                pitch_offsets=[-12, 12],
                velocity_offsets=[-4, 4],
                duration_offsets=[-0.5, 0.5],
            )

        print("Done.")


if __name__ == "__main__":
    main()