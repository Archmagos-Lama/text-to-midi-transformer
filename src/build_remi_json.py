import json
from pathlib import Path
from tqdm import tqdm
from miditok import REMI
from miditok.classes import TokenizerConfig
from miditoolkit import MidiFile  

RAW_MIDI_ROOT = "data/midi/"
OUT_JSON_DIR = "remi_json"

Path(OUT_JSON_DIR).mkdir(parents=True, exist_ok=True)

config = TokenizerConfig(
    use_programs=False,
    use_tempos=False,
    use_time_signatures=False,
    beat_res={(0, 4): 4},
    num_velocities=8,
    special_tokens=["BOS", "EOS"],
)

tokenizer = REMI(config)

tokenizer.save("data/tokenizer_config.json")

midi_files = list(Path(RAW_MIDI_ROOT).rglob("*.mid"))
midi_files += list(Path(RAW_MIDI_ROOT).rglob("*.midi"))

for mf in tqdm(midi_files):

    out_path = Path(OUT_JSON_DIR) / (mf.stem + ".json")
    if out_path.exists():
        continue

    try:
        midi = MidiFile(mf)

        midi.instruments = [
            inst for inst in midi.instruments
            if not inst.is_drum
        ]

        if len(midi.instruments) == 0:
            continue

        tokens = tokenizer(midi)
        tokenizer.save_tokens(tokens, out_path)

    except Exception as e:
        print("BAD:", mf)
        continue