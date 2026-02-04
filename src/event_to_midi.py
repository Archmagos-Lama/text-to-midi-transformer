import json
import pretty_midi

# ===================== CONFIG =====================

INPUT_EVENT_JSON = "data/generated_tokens.json"
OUTPUT_MIDI = "data/generated.mid"

# beat -> second（固定一个试听用 tempo）
BPM = 120
SEC_PER_BEAT = 60.0 / BPM

# ===================== INST -> PROGRAM (RENDER SKIN) =====================

INST_PROGRAM = {
    "INST_PIANO":   0,    # Acoustic Grand Piano
    "INST_LEAD":    81,   # Lead (saw)
    "INST_PAD":     89,   # Warm Pad
    "INST_STRINGS": 48,   # String Ensemble
    "INST_WIND":    73,   # Flute
    "INST_BASS":    33,   # Electric Bass
    "INST_PERC":    12,   # Marimba
    "INST_DRUM":    0,    # ignored (is_drum=True)
    "INST_FX":      99,
    "INST_OTHER":   0,
}

# ===================== LOAD EVENTS =====================

with open(INPUT_EVENT_JSON, "r", encoding="utf-8") as f:
    events = json.load(f)

# ===================== MIDI INIT =====================

midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
tracks = {}

def get_track(inst):
    if inst not in tracks:
        is_drum = (inst == "INST_DRUM")
        program = INST_PROGRAM.get(inst, 0)
        tracks[inst] = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=inst
        )
        midi.instruments.append(tracks[inst])
    return tracks[inst]

# ===================== PARSE EVENTS =====================

inst_time = {}
cur_inst = None
cur_pitch = None

for ev in events:
    if ev.startswith("INST_"):
        cur_inst = ev
        if cur_inst not in inst_time:
            inst_time[cur_inst] = 0.0
        continue

    if ev.startswith("NOTE_ON_"):
        cur_pitch = int(ev.split("_")[-1])
        continue

    if ev.startswith("DUR_"):
        if cur_inst is None or cur_pitch is None:
            continue

        dur = float(ev.split("_")[-1])

        start = inst_time[cur_inst] * SEC_PER_BEAT
        end   = (inst_time[cur_inst] + dur) * SEC_PER_BEAT

        track = get_track(cur_inst)
        track.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=cur_pitch,
                start=start,
                end=end
            )
        )

        inst_time[cur_inst] += dur
        cur_pitch = None


# ===================== WRITE MIDI =====================

midi.write(OUTPUT_MIDI)
print("Written MIDI:", OUTPUT_MIDI)
