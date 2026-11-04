"""
REMI tokenization with BARPOS + HarmRole v2 conditioning.

Improvement over v1: "non" is split into:
  - HarmRole_scale    : note is in the detected key's scale but not the chord
  - HarmRole_chromatic: note is outside the key entirely

This reduces the dominant "non" class (73.8% in v1) and gives the model
a richer harmonic signal.

Strategy:
  1. Run music21 on 422 originals → {stem: {"bars": bar_map, "scale": scale_pcs}}
  2. For augmented files, transpose both bar_map and scale_pcs by the semitone shift
  3. Inject BARPOS_k + HarmRole_Y (6 categories) after each Pitch_X
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
OUT_DIR   = Path("dataset_remi_harmrole_v2")
OUT_DIR.mkdir(exist_ok=True)

MAX_BARS   = 128
HARM_ROLES = [
    "HarmRole_root", "HarmRole_third", "HarmRole_fifth", "HarmRole_seventh",
    "HarmRole_scale",     # in-key passing/neighbor tone
    "HarmRole_chromatic", # outside key
]

# ── tokenizer ─────────────────────────────────────────────────────────────────
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
print(f"HarmRole   : {harm_ids}")

# ── scale intervals ────────────────────────────────────────────────────────────
MAJOR_IVS = [0, 2, 4, 5, 7, 9, 11]
MINOR_IVS = [0, 2, 3, 5, 7, 8, 10]  # natural minor

ROLE_BY_IV = {
    0: "root",
    2: "third", 3: "third", 4: "third",
    5: "fifth", 6: "fifth", 7: "fifth", 8: "fifth",
    9: "seventh", 10: "seventh", 11: "seventh",
}

def build_chord_map(m21_chord_obj):
    try:
        root_pc = m21_chord_obj.root().pitchClass
    except Exception:
        return None
    return {p.pitchClass: ROLE_BY_IV.get((p.pitchClass - root_pc) % 12, "chromatic")
            for p in m21_chord_obj.pitches}

def get_scale_pcs(score):
    """Return frozenset of pitch classes in the detected key."""
    try:
        k = score.analyze('key')
        root_pc = k.tonic.pitchClass
        ivs = MAJOR_IVS if k.mode == 'major' else MINOR_IVS
        return frozenset((root_pc + i) % 12 for i in ivs)
    except Exception:
        return frozenset(range(12))  # fallback: all pcs = no chromatic distinction

# ── music21 bar-level analysis ─────────────────────────────────────────────────
def get_bar_chord_map(midi_path):
    """Returns {"bars": {bar_idx: {pc: role}}, "scale": list_of_pcs}."""
    try:
        score      = converter.parse(str(midi_path))
        scale_pcs  = get_scale_pcs(score)
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
        return {"bars": bar_map, "scale": sorted(scale_pcs)}
    except Exception:
        return {"bars": {}, "scale": list(range(12))}

def transpose_entry(entry, semitones):
    if semitones == 0:
        return entry
    new_bars = {
        bar: {(pc + semitones) % 12: role for pc, role in chord.items()}
        for bar, chord in entry["bars"].items()
    }
    new_scale = [(pc + semitones) % 12 for pc in entry["scale"]]
    return {"bars": new_bars, "scale": new_scale}

# ── build cache ───────────────────────────────────────────────────────────────
TR_RE = re.compile(r"_tr([+-]\d+)$")

def orig_stem_and_shift(augmented_path):
    stem = augmented_path.stem
    m = TR_RE.search(stem)
    if m:
        return stem[:m.start()], int(m.group(1))
    return stem, 0

CACHE_FILE = OUT_DIR / "chord_cache_v2.json"

print("Step 1: analysing original MIDIs with music21 ...")
if CACHE_FILE.exists():
    with open(CACHE_FILE) as f:
        raw = json.load(f)
    chord_cache = {}
    for stem, entry in raw.items():
        chord_cache[stem] = {
            "bars":  {int(k): {int(pc): role for pc, role in chord.items()}
                      for k, chord in entry["bars"].items()},
            "scale": entry["scale"],
        }
    print(f"  loaded chord_cache from {CACHE_FILE} ({len(chord_cache)} originals)")
else:
    orig_files = list(ORIG_DIR.rglob("*.mid")) + list(ORIG_DIR.rglob("*.midi"))
    chord_cache = {}
    for mf in tqdm(orig_files, desc="music21"):
        chord_cache[mf.stem] = get_bar_chord_map(mf)
    with open(CACHE_FILE, "w") as f:
        json.dump(chord_cache, f)
    with_chords = sum(1 for v in chord_cache.values() if v["bars"])
    print(f"  cached {len(chord_cache)} originals, {with_chords} with chords")

# ── injection ──────────────────────────────────────────────────────────────────
def inject(ids, entry):
    bar_map   = entry["bars"]
    scale_set = set(entry["scale"])
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
            if chord_map and pc in chord_map:
                role = chord_map[pc]   # root/third/fifth/seventh
            elif pc in scale_set:
                role = "scale"
            else:
                role = "chromatic"
            out.append(harm_ids[f"HarmRole_{role}"])
        else:
            out.append(tid)
    return out

# ── tokenize ──────────────────────────────────────────────────────────────────
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
            orig_entry  = chord_cache.get(stem, {"bars": {}, "scale": list(range(12))})
            entry       = transpose_entry(orig_entry, shift)
            if not orig_entry["bars"]:
                cache_miss += 1

            midi    = MidiFile(midi_path)
            tok_seq = tokenizer(midi)
            if not tok_seq.ids:
                skipped += 1
                continue

            ids = [bos_id] + inject(tok_seq.ids, entry) + [eos_id]
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

# ── report role distribution ───────────────────────────────────────────────────
from collections import Counter
ids_list = train_tokens.tolist()
role_counts = Counter()
for tid in ids_list:
    for name, hid in harm_ids.items():
        if tid == hid:
            role_counts[name] += 1
total = sum(role_counts.values())
print("\nHarmRole distribution in train tokens:")
for r in HARM_ROLES:
    cnt = role_counts.get(r, 0)
    print(f"  {r:22s}: {cnt:8d}  ({100*cnt/total:.1f}%)")

meta = {
    "vocab_size":        tokenizer.vocab_size,
    "bar_pos_start":     bar_pos_start,
    "max_bars":          MAX_BARS,
    "harm_role_ids":     {k: int(v) for k, v in harm_ids.items()},
    "barpos_injected":   True,
    "harmrole_injected": True,
    "harmrole_version":  2,
}
with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone — vocab={tokenizer.vocab_size}")
print(f"Train: {len(train_tokens):,} -> {OUT_DIR}/train_tokens.pt")
print(f"Val:   {len(val_tokens):,}   -> {OUT_DIR}/val_tokens.pt")
