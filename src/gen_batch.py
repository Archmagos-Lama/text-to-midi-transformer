"""
Batch MIDI generation script. Usage:
  python src/gen_batch.py barpos 10 20   # seeds 10..19 -> generated_midis/
  python src/gen_batch.py baseline 9 20  # seeds 9..19  -> generated_midis_baseline/
"""
import sys, json
from pathlib import Path
import torch
import torch.nn.functional as F

mode    = sys.argv[1]          # "barpos" or "baseline"
seed_lo = int(sys.argv[2])
seed_hi = int(sys.argv[3])

if mode == "barpos":
    DATA_DIR  = Path("dataset_remi")
    MODEL_PATH = "runs_touhou_scratch/best.pt"
    TOK_PATH   = "dataset_remi/tokenizer.json"
    OUT_DIR    = Path("generated_midis")
else:
    DATA_DIR  = Path("dataset_remi_baseline")
    MODEL_PATH = "runs_touhou_baseline/best.pt"
    TOK_PATH   = "dataset_remi_baseline/tokenizer.json"
    OUT_DIR    = Path("generated_midis_baseline")

OUT_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "meta.json") as f:
    meta = json.load(f)
vocab_size    = meta["vocab_size"]
bar_pos_start = meta["bar_pos_start"]
max_bars      = meta["max_bars"]
barpos_injected = meta.get("barpos_injected", True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model definition (copied to avoid importing training script) ----
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(n_embd, 4 * n_embd)
        self.fc2  = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp  = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=8, n_head=8, n_embd=640, block_size=1536, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        x    = self.ln_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# ---- load ----
model = GPT(vocab_size=vocab_size).to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded {MODEL_PATH}  vocab={vocab_size}")

from miditok import REMI
tokenizer = REMI(params=TOK_PATH)

block_size        = 1536
max_tokens        = 4096
temperature       = 0.85
top_p             = 0.9
repetition_penalty = 1.15
recent_window     = 64

pitch_ids      = {i for i, s in enumerate(tokenizer.vocab) if s.startswith("Pitch_")}
position_ids   = {i for i, s in enumerate(tokenizer.vocab) if s.startswith("Position_")}
position_values = {i: int(s.split("_")[1]) for i, s in enumerate(tokenizer.vocab) if s.startswith("Position_")}
bar_pos_range  = set(range(bar_pos_start, bar_pos_start + max_bars))
bar_none_id    = tokenizer["Bar_None"]
bos_id         = tokenizer["BOS_None"]
eos_id         = tokenizer["EOS_None"]

def sample_next(logits, tokens):
    logits = logits[:, -1, :] / temperature
    for i in set(tokens[-recent_window:]):
        if i in pitch_ids:
            logits[:, i] /= repetition_penalty
    probs = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cp = torch.cumsum(sp, dim=-1)
    mask = cp > top_p; mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = False
    sp[mask] = 0; sp /= sp.sum()
    if tokens and tokens[-1] in position_ids:
        lp = position_values[tokens[-1]]
        for idx, t in enumerate(si[0]):
            if int(t) in position_ids and abs(position_values[int(t)] - lp) > 8:
                sp[0, idx] *= 0.1
        sp /= sp.sum()
    return int(si[0][torch.multinomial(sp[0], 1)])

def to_decode(tokens):
    if not barpos_injected:
        return tokens
    return [bar_none_id if t in bar_pos_range else t for t in tokens]

def generate(seed):
    torch.manual_seed(seed)
    tokens = [bos_id]
    for _ in range(max_tokens):
        x = torch.tensor(tokens[-block_size:], device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
        nt = sample_next(logits, tokens)
        tokens.append(nt)
        if nt == eos_id:
            break
    return tokens

for i in range(seed_lo, seed_hi):
    tokens = generate(i)
    midi   = tokenizer.decode(to_decode(tokens))
    p = OUT_DIR / f"full_{i:02d}.mid"
    midi.dump_midi(str(p))
    print(f"[seed {i:02d}] {p}  ({len(tokens)} tokens)")

print("Done")
