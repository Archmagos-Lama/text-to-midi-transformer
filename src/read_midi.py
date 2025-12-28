import pretty_midi
import collections
import argparse
import bisect

# ===================== CONFIG =====================

INPUT_MIDI_PATH  = "data/test_midis/15. Hartmann's Youkai Girl (sagittarius shikoku).mid"
OUTPUT_MIDI_PATH = "data/quantized/15_hartmann_quantized_1_32.mid"

GRID_BEAT = 0.125   # 1/32 beat
# ===== Load MIDI =====
midi = pretty_midi.PrettyMIDI(INPUT_MIDI_PATH)

print("\nTime signature changes:")
for ts in midi.time_signature_changes:
    print(f"  {ts.numerator}/{ts.denominator} at {ts.time:.3f}s")

print("Tempo changes:", midi.get_tempo_changes())
print("Number of instruments:", len(midi.instruments))

# ===== Beat timeline (tempo-aware) =====
beat_times = midi.get_beats()
GRID_BEAT = 0.125  # 1/32

# ===== Quantization =====
def time_to_beat(t, beat_times):
    """
    将秒时间映射为连续 beat（支持 tempo change）
    """
    if t <= beat_times[0]:
        return 0.0
    if t >= beat_times[-1]:
        return float(len(beat_times) - 1)

    idx = bisect.bisect_right(beat_times, t) - 1
    t0 = beat_times[idx]
    t1 = beat_times[idx + 1]
    # 在线性区间内插值
    return idx + (t - t0) / (t1 - t0)

def beat_to_time(b, beat_times):
    """
    将连续 beat 映射回秒时间
    """
    if b <= 0:
        return beat_times[0]
    if b >= len(beat_times) - 1:
        return beat_times[-1]

    i = int(b)
    frac = b - i
    return beat_times[i] + frac * (beat_times[i + 1] - beat_times[i])


def quantize_note(note, beat_times, grid_beat):
    start_beat = time_to_beat(note.start, beat_times)
    end_beat   = time_to_beat(note.end, beat_times)

    start_q = round(start_beat / grid_beat) * grid_beat
    end_q   = round(end_beat   / grid_beat) * grid_beat

    # 兜底：至少一个 grid
    if end_q <= start_q:
        end_q = start_q + grid_beat

    note.start = beat_to_time(start_q, beat_times)
    note.end   = beat_to_time(end_q,   beat_times)

for inst in midi.instruments:
    for note in inst.notes:
        quantize_note(note, beat_times, GRID_BEAT)

for idx, inst in enumerate(midi.instruments):
    if not inst.notes:
        continue

    pitches = [note.pitch for note in inst.notes]
    durations_sec = [note.end - note.start for note in inst.notes]

    print(f"\nInstrument {idx}")
    print(f"  Program: {inst.program}")
    print(f"  Is drum: {inst.is_drum}")
    print(f"  Notes: {len(inst.notes)}")
    print(f"  Pitch range: {min(pitches)} - {max(pitches)}")
    print(
        f"  Duration (sec): "
        f"min={min(durations_sec):.3f}, "
        f"max={max(durations_sec):.3f}, "
        f"mean={sum(durations_sec)/len(durations_sec):.3f}"
    )


# ===== Export quantized MIDI =====
quantized_midi = pretty_midi.PrettyMIDI()
quantized_midi.instruments = midi.instruments
quantized_midi.write(OUTPUT_MIDI_PATH)

print(f"\nQuantized MIDI written to: {OUTPUT_MIDI_PATH}")
