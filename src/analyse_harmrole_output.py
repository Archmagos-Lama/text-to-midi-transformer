"""
Analyse HarmRole token distribution in generated sequences.
Loads the harmrole tokenizer and decodes raw token ids from generated MIDIs
by re-tokenizing them, then counts HarmRole token frequencies.

Usage:
    python src/analyse_harmrole_output.py
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
from miditok import REMI
from miditoolkit import MidiFile
from tqdm import tqdm

TOK_PATH  = "dataset_remi_harmrole/tokenizer.json"
META_PATH = "dataset_remi_harmrole/meta.json"
GEN_DIR   = Path("generated_midis_harmrole")

tokenizer = REMI(params=TOK_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

harm_ids     = meta["harm_role_ids"]          # {'HarmRole_root': 132, ...}
bar_pos_start = meta["bar_pos_start"]
max_bars      = meta["max_bars"]
bar_none_id   = tokenizer["Bar_None"]
bar_pos_range = set(range(bar_pos_start, bar_pos_start + max_bars))

harm_id_set  = set(harm_ids.values())
harm_id_inv  = {v: k.replace("HarmRole_", "") for k, v in harm_ids.items()}

role_total   = Counter()
non_per_bar  = []   # fraction of non-chord tones per bar per file
consistency  = []   # fraction of bars where all notes share same chord role set

midi_files = sorted(GEN_DIR.glob("*.mid"))
print(f"Analysing {len(midi_files)} generated MIDIs ...")

for mf in tqdm(midi_files):
    try:
        midi    = MidiFile(mf)
        tok_seq = tokenizer(midi)
        ids     = tok_seq.ids
    except Exception as e:
        print(f"  skip {mf.name}: {e}")
        continue

    # We need to re-inject HarmRole tokens to analyse them.
    # Since generated MIDIs have HarmRole stripped, we can only analyse
    # the token sequence saved during generation — but we don't have that.
    # Instead, analyse the raw token stream for pitch class entropy
    # and compare against what the training data distribution looked like.
    #
    # Alternative: load train_tokens.pt and compare pitch patterns.
    pass

# ── Approach: re-tokenize generated MIDIs and compare pitch distribution
# against the chord-tone expectation from the training data.
# We compute: for each MIDI, what fraction of unique pitch classes
# appear in at most 3 clusters (indicating chord-like structure)?

from collections import Counter as C
import math

def pitch_class_chord_score(midi_path):
    """
    For each bar, collect pitch classes played.
    Score = fraction of bars where pitches cluster into ≤4 pitch classes
    (suggesting one chord per bar).
    """
    try:
        midi    = MidiFile(midi_path)
        tok_seq = tokenizer(midi)
    except:
        return None

    ids = tok_seq.ids
    bar_pitch_classes = defaultdict(set)
    bar_idx = 0
    bar_none = tokenizer["Bar_None"]

    for tid in ids:
        s = tokenizer[tid]
        if s == "Bar_None":
            bar_idx += 1
        elif s.startswith("Pitch_"):
            pc = int(s.split("_")[1]) % 12
            bar_pitch_classes[bar_idx].add(pc)

    if not bar_pitch_classes:
        return None

    chord_bars = sum(1 for pcs in bar_pitch_classes.values() if len(pcs) <= 5)
    return chord_bars / len(bar_pitch_classes), len(bar_pitch_classes)

# ── Load train_tokens.pt to get the HarmRole distribution in training data
print("\n── Training data HarmRole distribution ──")
try:
    train_tokens = torch.load("dataset_remi_harmrole/train_tokens.pt")
    ids = train_tokens.tolist()
    role_counts = Counter()
    for tid in ids:
        if tid in harm_id_set:
            role_counts[harm_id_inv[tid]] += 1
    total = sum(role_counts.values())
    for role in ["root", "third", "fifth", "seventh", "non"]:
        cnt = role_counts.get(role, 0)
        print(f"  {role:8s}: {cnt:8d}  ({100*cnt/total:.1f}%)")
    print(f"  chord tone rate (non-non): {100*(total-role_counts.get('non',0))/total:.1f}%")
except Exception as e:
    print(f"  could not load train_tokens.pt: {e}")

# ── Analyse pitch-class clustering in generated MIDIs
print("\n── Per-bar pitch clustering in generated MIDIs ──")
scores, bar_counts = [], []
for mf in tqdm(midi_files, desc="analysing"):
    result = pitch_class_chord_score(mf)
    if result:
        score, nb = result
        scores.append(score)
        bar_counts.append(nb)

if scores:
    print(f"  Files analysed : {len(scores)}")
    print(f"  Avg bars/file  : {sum(bar_counts)/len(bar_counts):.1f}")
    print(f"  Bars with ≤5 pitch classes: {100*sum(scores)/len(scores):.1f}%  (chord-like)")
    print(f"  Min / Max      : {min(scores):.2f} / {max(scores):.2f}")

# ── Compare with BARPOS generated MIDIs
print("\n── Per-bar pitch clustering in BARPOS generated MIDIs ──")
barpos_dir = Path("generated_midis")
barpos_tok  = REMI(params="dataset_remi/tokenizer.json")

def pitch_cluster_score_barpos(midi_path):
    try:
        midi    = MidiFile(midi_path)
        tok_seq = barpos_tok(midi)
    except:
        return None
    ids = tok_seq.ids
    bar_pitch_classes = defaultdict(set)
    bar_idx = 0
    for tid in ids:
        s = barpos_tok[tid]
        if s == "Bar_None":
            bar_idx += 1
        elif s.startswith("Pitch_"):
            pc = int(s.split("_")[1]) % 12
            bar_pitch_classes[bar_idx].add(pc)
    if not bar_pitch_classes:
        return None
    chord_bars = sum(1 for pcs in bar_pitch_classes.values() if len(pcs) <= 5)
    return chord_bars / len(bar_pitch_classes), len(bar_pitch_classes)

bp_scores = []
for mf in sorted(barpos_dir.glob("*.mid")):
    result = pitch_cluster_score_barpos(mf)
    if result:
        bp_scores.append(result[0])

if bp_scores:
    print(f"  Files analysed : {len(bp_scores)}")
    print(f"  Bars with ≤5 pitch classes: {100*sum(bp_scores)/len(bp_scores):.1f}%  (chord-like)")

# ── Reference MIDIs
print("\n── Per-bar pitch clustering in reference MIDIs (sample 50) ──")
import random, glob
ref_files = random.sample(glob.glob("data/touhou_midi_collection/**/*.mid", recursive=True), 50)
ref_scores = []
for mf in tqdm(ref_files, desc="reference"):
    result = pitch_cluster_score_barpos(mf)
    if result:
        ref_scores.append(result[0])
if ref_scores:
    print(f"  Files analysed : {len(ref_scores)}")
    print(f"  Bars with ≤5 pitch classes: {100*sum(ref_scores)/len(ref_scores):.1f}%  (chord-like)")
