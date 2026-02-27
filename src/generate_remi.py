import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# =================== model ===================

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.wte.weight

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= block_size

        pos = torch.arange(T, device=idx.device)

        # token + position embedding
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        # transformer blocks (no attn_mask anymore)
        for blk in self.blocks:
            x = blk(x)

        # final norm + head
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

        return logits, loss
    
# ===== config =====
CKPT_PATH = "data/pt/ckpt.pt"
device = "cuda"
dtype  = torch.bfloat16

vocab_size = 486
block_size = 1024
n_layer = 12
n_head  = 8
n_embd  = 512
dropout = 0.0

max_new = 2048
temperature = 0.9
top_p = 0.95

# ===== sampling =====

@torch.no_grad()
def sample(model, idx):
    for _ in range(max_new):
        min_id = idx.min().item()
        max_id = idx.max().item()

        assert 0 <= min_id, f"token < 0: {min_id}"
        assert max_id < vocab_size, f"token >= vocab_size: {max_id} >= {vocab_size}"
        
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)    
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cum > top_p] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_tok = torch.multinomial(sorted_probs, 1)
        next_tok = torch.gather(sorted_idx, -1, next_tok)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx

# ===== run =====

model = GPT().to(device)
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

prompt = torch.randint(0, vocab_size, (1, 1), device=device)
with torch.autocast("cuda", dtype=dtype):
    out = sample(model, prompt)

tokens = out[0].cpu().tolist()
torch.save(tokens, "gen_tokens.pt")
print(tokens)