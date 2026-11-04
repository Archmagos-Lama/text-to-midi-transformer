"""
REMI tokenization with BARPOS + HarmRole conditioning.

Strategy:
  1. Run music21 chordify() on the 422 original MIDIs only → bar_chord_maps cache
  2. For augmented files (_tr+N / _tr-N), transpose the cached chord map by N semitones
  3. MidiTok tokenize → inject BARPOS_k and HarmRole_Y after each Pitch_X

This runs music21 only 422 times instead of 4272 times.
"""
import json, os, re
from pathlib import Path

import torch
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from tqdm import tqdm
from music21 import converter, chord as m21chord, stream as m21stream

# ── paths ──────────────────────────────────────────────────────────────────────
ORIG_DIR  = Path(r"data/touhou_midi_collection")
TRAIN_DIR = Path(r"data/touhou_augmented_train")
VAL_DIR   = Path(r"data/touhou_augmented_val")
OUT_DIR   = Path("dataset_remi_harmrole")
OUT_DIR.mkdir(exist_ok=True)

MAX_BARS   = 128
HARM_ROLES = ["HarmRole_root", "HarmRole_third", "HarmRole_fifth",
              "HarmRole_seventh", "HarmRole_non"]

# ── tokenizer ─────────────────────────────────────────────────────────────────
# use_chords=False: MidiTok chord detection is ~5s/file and redundant —
# HarmRole tokens (from music21) already encode harmonic role information.
config = TokenizerConfig(
    use_chords=False, use_tempos=True, use_programs=True,
    use_time_signatures=False, use_sustain_pedals=False, use_pitch_bends=False,
    beat_res={(0, 4): 8, (4, 12): 4}, num_velocities=32,
    special_tokens=(
        ["PAD", "BOS", "EOS", "MASK"]
        + [f"BARPOS_{i}" for i in range(MAX_BARS)]
        + HARM_ROLES
    ),
)
tokenizer = REMI(config)
tokenizer.save(OUT_DIR / "tokenizer.json")

bar_none_id   = tokenizer["Bar_None"]
bar_pos_start = tokenizer["BARPOS_0"]
bos_id        = tokenizer["BOS_None"]
eos_id        = tokenizer["EOS_None"]
harm_ids      = {r: tokenizer[r] for r in HARM_ROLES}

print(f"Vocab size : {tokenizer.vocab_size}")

# ── interval → role ────────────────────────────────────────────────────────────
ROLE_BY_IV = {
    0: "root",
    2: "third", 3: "third", 4: "third",
    5: "fifth", 6: "fifth", 7: "fifth", 8: "fifth",
    9: "seventh", 10: "seventh", 11: "seventh",
}

def build_chord_map(m21_chord_obj):
    """Return {pitch_class: role_str} from a music21 Chord."""
    try:
        root_pc = m21_chord_obj.root().pitchClass
    except Exception:
        return None
    result = {}
    for p in m21_chord_obj.pitches:
        iv = (p.pitchClass - root_pc) % 12
        result[p.pitchClass] = ROLE_BY_IV.get(iv, "non")
    return result

# ── music21 bar-level analysis ─────────────────────────────────────────────────
def get_bar_chord_map(midi_path):
    """Returns {bar_index: {pitch_class: role_str}}. Empty dict on failure."""
    try:
        score      = converter.parse(str(midi_path))
        chordified = score.chordify()
        part = (chordified.parts[0]
                if isinstance(chordified, m21stream.Score)
                else chordified)
        bar_map = {}
        for i, measure in enumerate(part.getElementsByClass("Measure")):
            best, best_dur = None, 0.0
            for el in measure.flatten().getElementsByClass("Chord"):
                dur = float(el.duration.quarterLength)
                if dur > best_dur and len(el.pitches) >= 2:
                    best_dur, best = dur, el
            if best:
                cm = build_chord_map(best)
                if cm:
                    bar_map[i] = cm
        return bar_map
    except Exception:
        return {}

def transpose_chord_map(bar_map, semitones):
    """Shift all pitch classes by semitones mod 12."""
    if semitones == 0:
        return bar_map
    return {
        bar: {(pc + semitones) % 12: role for pc, role in chord.items()}
        for bar, chord in bar_map.items()
    }

# ── build cache: orig stem → bar_chord_map ────────────────────────────────────
TR_RE = re.compile(r"_tr([+-]\d+)$")

def orig_stem_and_shift(augmented_path):
    """
    'Song_tr+3.mid' → ('Song', 3)
    'Song_tr-2.mid' → ('Song', -2)
    'Song_tr+0.mid' → ('Song', 0)
    """
    stem = augmented_path.stem          # e.g. "Song_tr+3"
    m = TR_RE.search(stem)
    if m:
        return stem[:m.start()], int(m.group(1))
    return stem, 0

CACHE_FILE = OUT_DIR / "chord_cache.json"

print("Step 1: analysing original MIDIs with music21 ...")
if CACHE_FILE.exists():
    with open(CACHE_FILE) as f:
        raw = json.load(f)
    # JSON keys are strings; convert bar indices back to int
    chord_cache = {stem: {int(k): {int(pc): role for pc, role in chord.items()}
                           for k, chord in bar_map.items()}
                   for stem, bar_map in raw.items()}
    print(f"  loaded chord_cache from {CACHE_FILE} ({len(chord_cache)} originals)")
else:
    orig_files = list(ORIG_DIR.rglob("*.mid")) + list(ORIG_DIR.rglob("*.midi"))
    chord_cache = {}
    for mf in tqdm(orig_files, desc="music21"):
        bar_map = get_bar_chord_map(mf)
        chord_cache[mf.stem] = bar_map
    with open(CACHE_FILE, "w") as f:
        json.dump(chord_cache, f)
    print(f"  cached {len(chord_cache)} originals, "
          f"{sum(1 for v in chord_cache.values() if v)} with chords")

# ── injection ──────────────────────────────────────────────────────────────────
def inject(ids, bar_map):
    bar_count = 0
    out = []
    for tid in ids:
        ts = tokenizer[tid]
        if tid == bar_none_id:
            out.append(bar_pos_start + (bar_count % MAX_BARS))
            bar_count += 1
        elif ts.startswith("Pitch_"):
            out.append(tid)
            pc = int(ts.split("_")[1]) % 12
            chord_map = bar_map.get(bar_count, None)
            role = chord_map.get(pc, "non") if chord_map else "non"
            out.append(harm_ids[f"HarmRole_{role}"])
        else:
            out.append(tid)
    return out

# ── tokenize files ─────────────────────────────────────────────────────────────
def collect_midis(directory):
    files = []
    for root, _, fnames in os.walk(directory):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(Path(root) / f)
    return files

def tokenize_files(midi_files, label):
    all_ids, skipped, cache_miss = [], 0, 0
    for midi_path in tqdm(midi_files, desc=label):
        try:
            stem, shift = orig_stem_and_shift(midi_path)
            orig_map    = chord_cache.get(stem, {})
            bar_map     = transpose_chord_map(orig_map, shift)
            if not orig_map:
                cache_miss += 1

            midi    = MidiFile(midi_path)
            tok_seq = tokenizer(midi)
            if not tok_seq.ids:
                skipped += 1
                continue

            ids = [bos_id] + inject(tok_seq.ids, bar_map) + [eos_id]
            all_ids.extend(ids)
        except Exception as e:
            skipped += 1
            print(f"skip {midi_path.name}: {e}")
    print(f"{label}: {len(all_ids):,} tokens | skipped={skipped} | cache_miss={cache_miss}")
    return torch.tensor(all_ids, dtype=torch.long)

print("\nStep 2: tokenizing augmented files ...")
train_files = collect_midis(TRAIN_DIR)
val_files   = collect_midis(VAL_DIR)
print(f"Train MIDI: {len(train_files)}  Val MIDI: {len(val_files)}")

train_tokens = tokenize_files(train_files, "train")
val_tokens   = tokenize_files(val_files,   "val")

torch.save(train_tokens, OUT_DIR / "train_tokens.pt")
torch.save(val_tokens,   OUT_DIR / "val_tokens.pt")

meta = {
    "vocab_size":        tokenizer.vocab_size,
    "bar_pos_start":     bar_pos_start,
    "max_bars":          MAX_BARS,
    "harm_role_ids":     {k: int(v) for k, v in harm_ids.items()},
    "barpos_injected":   True,
    "harmrole_injected": True,
}
with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone — vocab={tokenizer.vocab_size}")
print(f"Train: {len(train_tokens):,} -> {OUT_DIR}/train_tokens.pt")
print(f"Val:   {len(val_tokens):,}   -> {OUT_DIR}/val_tokens.pt")
