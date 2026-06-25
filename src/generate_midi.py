import torch
from miditok import REMI
from pathlib import Path
from train_touhou_from_scratch import GPT

mode = input("mode (full / extend): ").strip().lower()

MODEL_PATH = "runs_touhou_scratch/best.pt"
TOKENIZER_PATH = "dataset_remi/tokenizer.json"

OUT_MIDI = "generated.mid"

device = "cuda" if torch.cuda.is_available() else "cpu"


import torch.nn as nn
import torch.nn.functional as F


block_size = 1536
vocab_size = 613
torch.seed()

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):

        B, T, C = x.size()

        qkv = self.qkv(x)

        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)

        self.ln2 = nn.LayerNorm(n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


model = GPT(vocab_size=613).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(ckpt["model"])

model.eval()

print("model loaded")

tokenizer = REMI(params=TOKENIZER_PATH)

print("tokenizer loaded")



max_tokens = 4096

temperature = 0.85
top_p = 0.9
repetition_penalty = 1.15

recent_window = 64
no_repeat_ngram = 8

pitch_ids = {i for i, s in enumerate(tokenizer.vocab) if s.startswith("Pitch_")}
position_ids = {i for i, s in enumerate(tokenizer.vocab) if s.startswith("Position_")}
position_values = {i: int(s.split("_")[1]) for i, s in enumerate(tokenizer.vocab) if s.startswith("Position_")}


def sample_next_token(logits, tokens):

    logits = logits[:, -1, :] / temperature

    # repetition penalty
    recent = set(tokens[-recent_window:])
    for i in recent:
        if i in pitch_ids:
            logits[:, i] /= repetition_penalty

    probs = torch.softmax(logits, dim=-1)

    # nucleus sampling
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum()

    candidates = sorted_idx[0]
    probs = sorted_probs[0]

    # no repeat ngram
    if len(tokens) > no_repeat_ngram:
        last_ngram = tokens[-(no_repeat_ngram - 1):]
        for i, t in enumerate(candidates):
            if tokens[-(no_repeat_ngram - 1):] == last_ngram:
                if int(t) in tokens[-no_repeat_ngram:]:
                    probs[i] *= 0.2

    # position continuity
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

    next_token = candidates[torch.multinomial(probs, 1)]

    return int(next_token)


# ---------- FULL ----------
if mode == "full":

    tokens = [tokenizer["Bar_None"]]

    for _ in range(max_tokens):

        x = torch.tensor(tokens[-block_size:], device=device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(x)

        next_token = sample_next_token(logits, tokens)
        tokens.append(next_token)

    midi = tokenizer.decode(tokens)
    midi.dump_midi(OUT_MIDI)

    print("saved:", OUT_MIDI)


# ---------- EXTEND ----------
elif mode == "extend":

    from symusic import Score

    midi_dir = Path("data/test_midis")
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))

    print("found", len(midi_files), "midis")

    for midi_path in midi_files:

        print("processing:", midi_path.name)

        score = Score(midi_path)
        tokens = tokenizer.encode(score).ids[:600]

        for _ in range(max_tokens):

            x = torch.tensor(tokens[-block_size:], device=device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(x)

            next_token = sample_next_token(logits, tokens)
            tokens.append(next_token)

        midi = tokenizer.decode(tokens)

        out_path = f"extended_{midi_path.stem}.mid"
        midi.dump_midi(out_path)

        print("saved:", out_path)