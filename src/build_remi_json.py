import json
from pathlib import Path
from tqdm import tqdm
from miditok import REMI
from miditok.classes import TokenizerConfig

RAW_MIDI_ROOT = "data/midi"
OUT_JSON_DIR = "remi_json"

Path(OUT_JSON_DIR).mkdir(parents=True, exist_ok=True)

config = TokenizerConfig(
    use_programs=True,
    use_tempos=True,
    use_time_signatures=True,
    beat_res={(0, 4): 8},
    num_velocities=32,
)

tokenizer = REMI(config)

midi_files = list(Path(RAW_MIDI_ROOT).rglob("*.mid"))
midi_files += list(Path(RAW_MIDI_ROOT).rglob("*.midi"))

bad = 0

for mf in tqdm(midi_files):

    out_path = Path(OUT_JSON_DIR) / (mf.stem + ".json")

    # 断点续跑：已存在则跳过
    if out_path.exists():
        continue

    try:
        tokens = tokenizer(mf)
        tokenizer.save_tokens(tokens, out_path)
    except Exception as e:
        bad += 1
        print("BAD:", mf)
        continue

print("Bad files:", bad)