import os
import bisect
import pretty_midi
import warnings
from itertools import groupby

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Tempo, Key or Time signature change events found"
)

# ===================== CONFIG =====================

INPUT_ROOTS = [
    "data/touhou_midi_collection"
]

OUT_MIDI_ROOT  = "data/quantized_midi"
OUT_EVENT_ROOT = "data/events"

GRID_BEAT = 0.125  # 1/32 beat

os.makedirs(OUT_MIDI_ROOT, exist_ok=True)
os.makedirs(OUT_EVENT_ROOT, exist_ok=True)

def make_out_path(in_path, in_root, out_root, new_ext=".txt"):
    """
    将 in_path 相对 in_root 的路径，原样映射到 out_root
    """
    rel = os.path.relpath(in_path, in_root)
    rel = os.path.splitext(rel)[0] + new_ext
    out_path = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


# ===================== INST -> PROGRAM (FOR MIDI RENDER ONLY) =====================

DEFAULT_PROGRAM_FOR_INST = {
    "INST_PIANO":   0,    # Acoustic Grand Piano
    "INST_LEAD":    81,   # Lead 2 (saw)
    "INST_PAD":     89,   # Warm Pad
    "INST_STRINGS": 48,   # String Ensemble
    "INST_WIND":    73,   # Flute
    "INST_BASS":    33,   # Electric Bass (finger)
    "INST_PERC":    12,   # Marimba
    "INST_DRUM":    0,    # ignored when is_drum=True
    "INST_FX":      99,   # Atmosphere
    "INST_OTHER":   0
}


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

# ===================== NORMALIZATION =====================

def normalize_time_sig(numer, denom):
    if numer == 1:
        return None

    if denom == 16 and numer % 2 == 0:
        numer //= 2
        denom = 8
    if denom == 8 and numer % 2 == 0:
        numer //= 2
        denom = 4

    if (numer, denom) in [
        (2,4), (3,4), (4,4),
        (6,8), (7,8)
    ]:
        return f"{numer}/{denom}"

    return "OTHER"

BASE_DURS = [
    # straight
    0.125, 0.25, 0.5,

    # swing 8th (≈2:1)
    1/3,        # ≈0.333
    2/3,        # ≈0.667

    # dotted / compound
    0.375,      # dotted 16th
    0.75,       # dotted 8th

    # beat-level
    1.0, 1.5, 2.0,
    3.0, 4.0,

    # long
    6.0, 8.0, 16.0
]


def bucket_duration(d):
    return min(BASE_DURS, key=lambda x: abs(x - d))

def map_instrument(inst):
    if inst.is_drum:
        return "INST_DRUM"

    p = inst.program

    # Piano / Keys（扩大）
    if p < 16:                     # Piano + EP + Clav
        return "INST_PIANO"

    # Organ / Keys（并入 Piano）
    if 16 <= p < 32:
        return "INST_PIANO"

    # Bass（保持）
    if 32 <= p < 40:
        return "INST_BASS"

    # Strings / Pads（扩大）
    if 40 <= p < 56:
        return "INST_STRINGS"
    if 88 <= p < 96:               # Synth Strings / Choir
        return "INST_PAD"

    # Brass / Wind / Lead（扩大）
    if 56 <= p < 72:
        return "INST_WIND"
    if 80 <= p < 88:               # Synth Lead
        return "INST_LEAD"

    # FX（明确分离）
    if 112 <= p < 128:
        return "INST_FX"

    return "INST_OTHER"



# ===================== IO UTILS =====================

def iter_midi_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".mid", ".midi")):
                yield os.path.join(dirpath, fn)

def rel_out_path(in_path, in_root, out_root, ext):
    rel = os.path.relpath(in_path, in_root)
    rel = os.path.splitext(rel)[0] + ext
    out = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out

# ===================== MAIN =====================

total_files = 0
bad_files   = 0

for root in INPUT_ROOTS:
    for midi_path in iter_midi_files(root):
        total_files += 1
        if total_files % 50 == 0:
            print(f"[quantize] processed {total_files}, bad {bad_files}")

            
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            bad_files += 1
            continue

        beat_times = midi.get_beats()
        if len(beat_times) < 2:
            bad += 1
            continue

        # ---------- quantize notes ----------
        for inst in midi.instruments:
            for note in inst.notes:
                sb = time_to_beat(note.start, beat_times)
                eb = time_to_beat(note.end,   beat_times)

                start_q = round(sb / GRID_BEAT) * GRID_BEAT
                dur_q   = quantize_duration_beat(eb - sb, GRID_BEAT)
                dur_q   = bucket_duration(dur_q)

                end_q = start_q + dur_q

                note.start = beat_to_time(start_q, beat_times)
                note.end   = beat_to_time(end_q,   beat_times)

                note._start_beat = start_q
                note._dur_beat   = dur_q

        # ---------- export quantized MIDI (INST-based, clean) ----------

        qm = pretty_midi.PrettyMIDI()
        qm.time_signature_changes = midi.time_signature_changes

        inst_tracks = {}

        def get_track(inst_evt):
            if inst_evt not in inst_tracks:
                is_drum = (inst_evt == "INST_DRUM")
                program = DEFAULT_PROGRAM_FOR_INST.get(inst_evt, 0)

                inst_tracks[inst_evt] = pretty_midi.Instrument(
                    program=program,
                    is_drum=is_drum,
                    name=inst_evt
                )
                qm.instruments.append(inst_tracks[inst_evt])
            return inst_tracks[inst_evt]

        # 收集所有量化后的 note，按 INST 写回
        for inst in midi.instruments:
            inst_evt = map_instrument(inst)
            tr = get_track(inst_evt)

            for note in inst.notes:
                tr.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end
                    )
                )

        out_midi = make_out_path(midi_path, root, OUT_MIDI_ROOT, ".mid")
        qm.write(out_midi)



        # ---------- export events ----------
        events = []

        # time sig events
        for ts in midi.time_signature_changes:
            norm = normalize_time_sig(ts.numerator, ts.denominator)
            if norm is not None:
                events.append(f"TIME_SIG_{norm.replace('/', '_')}")
        
        # default time sig
        if not any(e.startswith("TIME_SIG_") for e in events):
            events.append("TIME_SIG_4_4")

        notes = []
        for inst in midi.instruments:
            inst_evt = map_instrument(inst)
            for note in inst.notes:
                notes.append((
                    note._start_beat,
                    inst_evt,
                    note.pitch,
                    note._dur_beat
                ))

        notes.sort(key=lambda x: x[0])

        cur_inst = None

        for start_beat, group in groupby(notes, key=lambda x: x[0]):
            group = sorted(group, key=lambda x: x[1])

            for _, inst_evt, pitch, dur in group:
                if inst_evt != cur_inst:
                    events.append(inst_evt)
                    cur_inst = inst_evt
                events.append(f"NOTE_ON_{pitch}")
                events.append(f"DUR_{dur}")

        out_evt = make_out_path(midi_path,root,OUT_EVENT_ROOT,".txt")

        with open(out_evt, "w", encoding="utf-8") as f:
            for e in events:
                f.write(e + "\n")

print("==== DONE ====")
print("Total processed:", total_files)
print("Bad skipped   :", bad_files)
