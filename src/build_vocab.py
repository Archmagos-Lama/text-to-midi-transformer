import os
import bisect
import json
import pretty_midi
from collections import Counter
import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Tempo, Key or Time signature change events found"
)



# ===================== CONFIG =====================

DATASET_ROOTS = [
    "data/clean_midi",
    "data/touhou_midi_collection"
]

GRID_BEAT = 0.125   # 1/32 beat

OUT_TIME_SIG = "data/vocab/time_sig_vocab.txt"
OUT_DUR      = "data/vocab/dur_vocab.txt"
OUT_VOCAB    = "data/vocab/vocab.json"

bad_log = open("data/vocab/bad_midi_files.txt", "w", encoding="utf-8")


os.makedirs("data/vocab", exist_ok=True)

# ===================== QUANTIZATION UTILS =====================

def time_to_beat(t, bt):
    if t <= bt[0]: 
        return 0.0
    if t >= bt[-1]: 
        return float(len(bt) - 1)
    i = bisect.bisect_right(bt, t) - 1
    return i + (t - bt[i]) / (bt[i+1] - bt[i])

def beat_to_time(b, bt):
    if b <= 0: 
        return bt[0]
    if b >= len(bt) - 1: 
        return bt[-1]
    i = int(b)
    f = b - i
    return bt[i] + f * (bt[i+1] - bt[i])

def quantize_duration_beat(dur, grid):
    q = round(dur / grid) * grid
    return max(q, grid)

def map_instrument(inst):
    if inst.is_drum:
        return "INST_DRUM"
    p = inst.program
    if p < 8:
        return "INST_PIANO"
    if 32 <= p < 40:
        return "INST_BASS"
    if 40 <= p < 56:
        return "INST_STRINGS"
    if 72 <= p < 80:
        return "INST_WIND"
    return "INST_OTHER"

# ===================== COLLECT =====================

time_sig_set = set()
dur_counter  = Counter()

total_files = 0
bad_files   = 0

def iter_midi_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".mid", ".midi")):
                yield os.path.join(dirpath, fn)

total_files = 0
bad_files = 0
last_print = 0

for root in DATASET_ROOTS:
    for midi_path in iter_midi_files(root):
        total_files += 1

        # ===== 进度心跳（每 100 个文件）=====
        if total_files - last_print >= 100:
            print(f"[build_vocab] processed {total_files} files "
                  f"(bad: {bad_files})")
            last_print = total_files

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            bad_files += 1
            bad_log.write(midi_path + "\n")
            continue


        # ---- TIME SIG ----
        for ts in midi.time_signature_changes:
            time_sig_set.add(f"{ts.numerator}/{ts.denominator}")

        # ---- DUR (tempo-aware, quantized) ----
        beat_times = midi.get_beats()
        if len(beat_times) < 2:
            continue

        for inst in midi.instruments:
            for note in inst.notes:
                sb = time_to_beat(note.start, beat_times)
                eb = time_to_beat(note.end,   beat_times)

                start_q = round(sb / GRID_BEAT) * GRID_BEAT
                dur_q   = quantize_duration_beat(eb - sb, GRID_BEAT)

                dur_counter[dur_q] += 1


# ===================== WRITE RESULTS =====================

# ---- TIME SIG ----
with open(OUT_TIME_SIG, "w", encoding="utf-8") as f:
    for ts in sorted(time_sig_set):
        f.write(ts + "\n")

# ---- DUR ----
with open(OUT_DUR, "w", encoding="utf-8") as f:
    f.write("DUR(beat) count\n")
    for d, c in dur_counter.most_common():
        f.write(f"{d:<8} {c}\n")

# ---- FINAL VOCAB (FROZEN) ----
vocab = {}

idx = 0

# SPECIAL
for tok in ["<START>", "<END>"]:
    vocab[tok] = idx
    idx += 1

# TIME SIG
for ts in sorted(time_sig_set):
    vocab[f"TIME_SIG_{ts.replace('/', '_')}"] = idx
    idx += 1

# INST (固定，不来自统计)
for inst in [
    "INST_PIANO",
    "INST_BASS",
    "INST_STRINGS",
    "INST_WIND",
    "INST_DRUM",
    "INST_OTHER"
]:
    vocab[inst] = idx
    idx += 1

# NOTE_ON（固定 0–127）
for p in range(128):
    vocab[f"NOTE_ON_{p}"] = idx
    idx += 1

# DUR（来自统计）
for d in sorted(dur_counter.keys()):
    vocab[f"DUR_{d}"] = idx
    idx += 1

with open(OUT_VOCAB, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

# ===================== REPORT =====================

print("==== build_vocab finished ====")
print(f"Total MIDI files scanned: {total_files}")
print(f"Bad MIDI files skipped : {bad_files}")
print(f"Unique TIME_SIG        : {len(time_sig_set)}")
print(f"Unique DUR             : {len(dur_counter)}")
print(f"Total vocab size       : {len(vocab)}")

print("Written:")
print(" ", OUT_TIME_SIG)
print(" ", OUT_DUR)
print(" ", OUT_VOCAB)
