import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_PATH     = Path("dataset_remi_baseline/train_tokens.pt")
VAL_PATH       = Path("dataset_remi_baseline/val_tokens.pt")
META_JSON = Path("dataset_remi_baseline/meta.json")
OUT_DIR   = Path("runs_touhou_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(META_JSON) as f:
    _meta = json.load(f)
vocab_size = _meta["vocab_size"]

block_size   = 1536
batch_size   = 12
grad_accum   = 8

max_iters    = 60000

lr           = 3e-4
warmup_iters = 2000
min_lr       = 3e-5

eval_interval = 500
eval_iters    = 50
log_interval  = 50

save_interval     = 2000
save_latest_every = 200

seed = 1337
torch.manual_seed(seed)

device    = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float16 if device == "cuda" else torch.float32

LATEST_CKPT = OUT_DIR / "latest.pt"

train_tokens = torch.load(TRAIN_PATH).long()
val_tokens   = torch.load(VAL_PATH).long()

N_train = train_tokens.numel()
N_val   = val_tokens.numel()

for name, tokens in [("train", train_tokens), ("val", val_tokens)]:
    mx = int(tokens.max()) + 1
    if mx > vocab_size:
        raise RuntimeError(f"{name} dataset has token id >= vocab_size ({mx} > {vocab_size})")

print(f"Train tokens: {N_train}  Val tokens: {N_val}")


def get_batch(split="train"):
    tokens = train_tokens if split == "train" else val_tokens
    N = tokens.numel()
    ix = torch.randint(0, N - block_size - 1, (batch_size,))
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.dropout  = dropout

        self.qkv       = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj      = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


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
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=8, n_head=8, n_embd=640, dropout=0.1):
        super().__init__()
        self.n_layer = n_layer
        self.n_head  = n_head
        self.n_embd  = n_embd

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


model     = GPT(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


def get_lr(it):
    if it < warmup_iters:
        return lr * it / warmup_iters
    t   = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + cos * (lr - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                _, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def train():
    start_iter     = 0
    best_eval_loss = None

    if LATEST_CKPT.exists():
        print(f"[resume] loading {LATEST_CKPT}")
        ckpt = torch.load(LATEST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and device == "cuda":
            scaler.load_state_dict(ckpt["scaler"])
        start_iter     = int(ckpt.get("iter", 0))
        best_eval_loss = ckpt.get("best_eval_loss", None)
        print(f"[resume] start_iter = {start_iter}")

    t0      = time.time()
    running = 0.0

    for it in range(start_iter + 1, max_iters + 1):
        lr_now = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        for micro in range(grad_accum):
            x, y = get_batch("train")
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            running += loss.item()

        scaler.step(optimizer)
        scaler.update()

        if it % log_interval == 0:
            dt      = time.time() - t0
            avg_loss = running / log_interval
            running  = 0.0
            print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {dt:.1f}s")
            t0 = time.time()

        if it % eval_interval == 0:
            losses = estimate_loss()
            print(f"[eval] iter {it:6d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

            if best_eval_loss is None or losses["val"] < best_eval_loss:
                best_eval_loss = losses["val"]
                best_path = OUT_DIR / "best.pt"
                torch.save(
                    {
                        "model":          model.state_dict(),
                        "optimizer":      optimizer.state_dict(),
                        "scaler":         scaler.state_dict() if device == "cuda" else None,
                        "iter":           it,
                        "best_eval_loss": best_eval_loss,
                        "config": {
                            "vocab_size":  vocab_size,
                            "block_size":  block_size,
                            "batch_size":  batch_size,
                            "grad_accum":  grad_accum,
                        },
                    },
                    best_path,
                )
                print(f"[save] best -> {best_path}  (val loss {best_eval_loss:.4f})")

        if it % save_latest_every == 0:
            torch.save(
                {
                    "model":          model.state_dict(),
                    "optimizer":      optimizer.state_dict(),
                    "scaler":         scaler.state_dict() if device == "cuda" else None,
                    "iter":           it,
                    "best_eval_loss": best_eval_loss,
                    "config": {
                        "vocab_size":  vocab_size,
                        "block_size":  block_size,
                        "batch_size":  batch_size,
                        "grad_accum":  grad_accum,
                    },
                },
                LATEST_CKPT,
            )

        if it % save_interval == 0:
            out = OUT_DIR / f"ckpt_{it}.pt"
            torch.save(
                {
                    "model":          model.state_dict(),
                    "optimizer":      optimizer.state_dict(),
                    "scaler":         scaler.state_dict() if device == "cuda" else None,
                    "iter":           it,
                    "best_eval_loss": best_eval_loss,
                    "config": {
                        "vocab_size":  vocab_size,
                        "block_size":  block_size,
                        "batch_size":  batch_size,
                        "grad_accum":  grad_accum,
                    },
                },
                out,
            )
            print(f"[save] {out}")

    print("[done]")


if __name__ == "__main__":
    train()
