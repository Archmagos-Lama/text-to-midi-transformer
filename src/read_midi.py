import pretty_midi
import bisect
import collections

# ===================== PATHS =====================
INPUT_MIDI  = "data/test_midis/15. Hartmann's Youkai Girl (sagittarius shikoku).mid"
OUTPUT_MIDI = "data/quantized/15_hartmann_quantized_1_32.mid"
OUTPUT_DUR  = "data/quantized/15_hartmann_dur.txt"
OUTPUT_EVT  = "data/quantized/15_hartmann_events.txt"

# ===================== PARAMS =====================
GRID_BEAT = 0.125  # 1/32 beat

# ===================== LOAD ======================
midi = pretty_midi.PrettyMIDI(INPUT_MIDI)
beat_times = midi.get_beats()

print("Tempo changes:", midi.get_tempo_changes())
print("Time signature changes:")
for ts in midi.time_signature_changes:
    print(f"  {ts.numerator}/{ts.denominator} at {ts.time:.3f}s")
print("Instruments:", len(midi.instruments))

# ===================== UTIL ======================
def time_to_beat(t, bt):
    if t <= bt[0]: return 0.0
    if t >= bt[-1]: return float(len(bt) - 1)
    i = bisect.bisect_right(bt, t) - 1
    return i + (t - bt[i]) / (bt[i+1] - bt[i])

def beat_to_time(b, bt):
    if b <= 0: return bt[0]
    if b >= len(bt) - 1: return bt[-1]
    i = int(b)
    f = b - i
    return bt[i] + f * (bt[i+1] - bt[i])

def quantize_duration_beat(dur, grid):
    q = round(dur / grid) * grid
    return max(q, grid)  # 至少一个 grid

def map_instrument(inst):
    if inst.is_drum: return "INST_DRUM"
    p = inst.program
    if p < 8: return "INST_PIANO"
    if 40 <= p < 56: return "INST_STRINGS"
    if 72 <= p < 80: return "INST_WIND"
    return "INST_OTHER"

# ===================== QUANTIZE ==================
for inst in midi.instruments:
    for note in inst.notes:
        sb = time_to_beat(note.start, beat_times)
        eb = time_to_beat(note.end,   beat_times)

        start_q = round(sb / GRID_BEAT) * GRID_BEAT
        dur_q   = quantize_duration_beat(eb - sb, GRID_BEAT)
        end_q   = start_q + dur_q

        note.start = beat_to_time(start_q, beat_times)
        note.end   = beat_to_time(end_q,   beat_times)

        # ★ 权威 duration（beat 域），供后续使用
        note._dur_beat = dur_q
        note._start_beat_q = start_q

# ===================== EXPORT MIDI ===============
qm = pretty_midi.PrettyMIDI()
qm.instruments = midi.instruments
qm.write(OUTPUT_MIDI)
print("Quantized MIDI:", OUTPUT_MIDI)

# ===================== EXPORT DUR =================
dur_counter = collections.Counter()
for inst in midi.instruments:
    for note in inst.notes:
        dur_counter[note._dur_beat] += 1

with open(OUTPUT_DUR, "w", encoding="utf-8") as f:
    f.write("DUR (beat)  count\n")
    for d, c in dur_counter.most_common():
        f.write(f"{d:<8} {c}\n")

print("DUR list:", OUTPUT_DUR)

# ===================== EXPORT EVENTS ==============
# collect time-sig events on beat timeline
ts_events = []
for ts in midi.time_signature_changes:
    b = time_to_beat(ts.time, beat_times)
    ts_events.append((b, f"TIME_SIG_{ts.numerator}_{ts.denominator}"))
ts_events.sort(key=lambda x: x[0])

# collect notes
notes = []
for inst in midi.instruments:
    inst_evt = map_instrument(inst)
    for note in inst.notes:
        notes.append((
            note._start_beat_q,
            inst_evt,
            note.pitch,
            note._dur_beat
        ))
notes.sort(key=lambda x: x[0])

events = []
ti = 0
cur_inst = None
for sb, inst_evt, pitch, dur in notes:
    while ti < len(ts_events) and ts_events[ti][0] <= sb:
        events.append(ts_events[ti][1])
        ti += 1
    if inst_evt != cur_inst:
        events.append(inst_evt)
        cur_inst = inst_evt
    events.append(f"NOTE_ON_{pitch}")
    events.append(f"DUR_{dur}")

with open(OUTPUT_EVT, "w", encoding="utf-8") as f:
    for e in events:
        f.write(e + "\n")

print("Event doc:", OUTPUT_EVT)
print("Total events:", len(events))
