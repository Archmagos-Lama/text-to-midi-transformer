import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# allow importing from same directory
sys.path.insert(0, str(Path(__file__).parent))
from train_touhou_baseline import GPT

MODEL_PATH     = "runs_touhou_baseline/best.pt"
TOKENIZER_PATH = "dataset_remi_baseline/tokenizer.json"
OUT_MIDI       = "generated.mid"

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("dataset_remi_baseline/meta.json") as f:
    _meta = json.load(f)
vocab_size    = _meta["vocab_size"]
bar_pos_start = _meta["bar_pos_start"]
max_bars      = _meta["max_bars"]
block_size = 1536

torch.seed()

model = GPT(vocab_size=vocab_size).to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print("model loaded  vocab_size:", vocab_size)

from miditok import REMI
tokenizer = REMI(params=TOKENIZER_PATH)
print("tokenizer loaded")

max_tokens        = 4096
temperature       = 0.85
top_p             = 0.9
repetition_penalty = 1.15
recent_window     = 64
no_repeat_ngram   = 8

pitch_ids      = {i for i, s in enumerate(tokenizer.vocab) if s.startswith("Pitch_")}
position_ids   = {i for i, s in enumerate(tokenizer.vocab) if s.startswith("Position_")}
position_values = {i: int(s.split("_")[1]) for i, s in enumerate(tokenizer.vocab) if s.startswith("Position_")}


def sample_next_token(logits, tokens):
    logits = logits[:, -1, :] / temperature

    recent = set(tokens[-recent_window:])
    for i in recent:
        if i in pitch_ids:
            logits[:, i] /= repetition_penalty

    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0]  = False
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum()

    candidates = sorted_idx[0]
    probs      = sorted_probs[0]

    if len(tokens) > no_repeat_ngram:
        last_ngram = tokens[-(no_repeat_ngram - 1):]
        for i, t in enumerate(candidates):
            if tokens[-(no_repeat_ngram - 1):] == last_ngram:
                if int(t) in tokens[-no_repeat_ngram:]:
                    probs[i] *= 0.2

    if tokens:
        last_id = tokens[-1]
        if last_id in position_ids:
            last_pos = position_values[last_id]
            for i, t in enumerate(candidates):
                tid = int(t)
                if tid in position_ids:
                    new_pos = position_values[tid]
                    if abs(new_pos - last_pos) > 8:
                        probs[i] *= 0.1

    probs /= probs.sum()
    return int(candidates[torch.multinomial(probs, 1)])


bar_none_id   = tokenizer["Bar_None"]
bar_pos_range = set(range(bar_pos_start, bar_pos_start + max_bars))
bos_id        = tokenizer["BOS_None"]
eos_id        = tokenizer["EOS_None"]


def to_decode_ids(tokens):
    """Replace BARPOS_X back to Bar_None so miditok decoder understands bar boundaries."""
    return [bar_none_id if t in bar_pos_range else t for t in tokens]


def generate(prompt_ids, n_new):
    tokens = list(prompt_ids)
    for _ in range(n_new):
        x = torch.tensor(tokens[-block_size:], device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
        next_tok = sample_next_token(logits, tokens)
        tokens.append(next_tok)
        if next_tok == eos_id:
            break
    return tokens


mode = input("mode (full / full_batch / extend): ").strip().lower()

# ---------- FULL ----------
if mode == "full":
    tokens = generate([bos_id], max_tokens)
    midi = tokenizer.decode(to_decode_ids(tokens))
    midi.dump_midi(OUT_MIDI)
    print("saved:", OUT_MIDI)

# ---------- FULL BATCH ----------
elif mode == "full_batch":
    n = int(input("how many MIDIs to generate: ").strip())
    out_dir = Path("generated_midis_baseline")
    out_dir.mkdir(exist_ok=True)
    for i in range(n):
        torch.manual_seed(i)
        tokens = generate([bar_pos_start], max_tokens)
        midi = tokenizer.decode(to_decode_ids(tokens))
        out_path = out_dir / f"full_{i:02d}.mid"
        midi.dump_midi(str(out_path))
        print(f"  [{i+1}/{n}] saved: {out_path}")

# ---------- EXTEND ----------
elif mode == "extend":
    from symusic import Score

    midi_dir   = Path("data/test_midis")
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    print("found", len(midi_files), "midis")

    out_dir = Path("generated_midis_baseline")
    out_dir.mkdir(exist_ok=True)
    for midi_path in midi_files:
        print("processing:", midi_path.name)
        score   = Score(midi_path)
        raw_ids = tokenizer.encode(score).ids[:600]

        bar_count = 0
        prompt = [bos_id]
        for tid in raw_ids:
            if tid == bar_none_id:
                prompt.append(bar_pos_start + (bar_count % max_bars))
                bar_count += 1
            else:
                prompt.append(tid)

        tokens   = generate(prompt, max_tokens)
        midi     = tokenizer.decode(to_decode_ids(tokens))
        out_path = out_dir / f"extended_{midi_path.stem}.mid"
        midi.dump_midi(str(out_path))
        print("saved:", out_path)
