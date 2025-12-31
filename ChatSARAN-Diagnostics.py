#!/usr/bin/env python3

"""
Diagnostic script for ChatSARAN.

Run this in the same environment as your training output to print:
 - greedy decode sample
 - top-k probabilities for the last token
 - token embedding gradient norm (one backward step)
 - model parameter count

Usage:
  python ChatSARAN-Diagnostics.py
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# adjust paths if needed
DATA_IDS = "data_ids.pt"
CKPT_PATH = "chat_saran_ckpt.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")

if not os.path.isfile(CKPT_PATH):
    print("Checkpoint not found:", CKPT_PATH)
    sys.exit(1)
if not os.path.isdir(TOKENIZER_DIR):
    print("Tokenizer dir not found:", TOKENIZER_DIR)
    sys.exit(1)
if not os.path.isfile(DATA_IDS):
    print("data_ids.pt not found:", DATA_IDS)
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
tokenizer.model_max_length = 10**6
eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

# Recreate the single-block ChatSARAN (matches inference/training - residual scaling included)
class ChatSARAN(nn.Module):
    def __init__(self, vocab_size, d_model=384, block_size=256, residual_scale=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.residual_scale = float(residual_scale)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model), dtype=torch.float32))
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
        self.Wout = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, idx, use_last_token=False):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))
        x = tok + pos
        q = self.Wq(x); k = self.Wk(x); v = self.Wv(x)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        m = self.mask[:T, :T].to(scores.device)
        scores = scores.masked_fill(~m, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_out = attn @ v
        x = x + self.residual_scale * attn_out
        last = x[:, -1, :]
        if use_last_token:
            return self.Wout(last)
        return self.Wout(x)

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int = 64, temperature: float = None, top_k: int = None):
        """
        Simple generation loop (greedy or sampled depending on temperature/top_k).
        """
        temperature = 1.0 if temperature is None else float(temperature)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:].to(next(self.parameters()).device)
            logits = self(idx_cond, use_last_token=True)  # (B, V)

            # greedy
            if temperature is not None and float(temperature) <= 0.0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits_proc = logits / temperature if temperature is not None else logits
                if top_k is not None:
                    k = min(top_k, logits_proc.size(-1))
                    v, _ = torch.topk(logits_proc, k)
                    cutoff = v[:, -1].unsqueeze(1)
                    logits_proc = torch.where(logits_proc < cutoff, torch.full_like(logits_proc, float("-1e10")), logits_proc)
                probs = F.softmax(logits_proc, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx.to(next(self.parameters()).device), nxt], dim=1)
        return idx

# load checkpoint
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
d_model = ckpt.get("d_model", 384)
block_size = ckpt.get("block_size", 256)
residual_scale = ckpt.get("residual_scale", 0.1)
model = ChatSARAN(ckpt["vocab_size"], d_model=d_model, block_size=block_size, residual_scale=residual_scale).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

print("Loaded model:", sum(p.numel() for p in model.parameters()), "params")
print("Residual scale:", residual_scale)
print("Device:", DEVICE)

# load data ids and make a minibatch for grad check
ids = torch.load(DATA_IDS)
data = torch.tensor(ids, dtype=torch.long)
n_train = int(len(data) * 0.95)
train_data = data[:n_train]
val_data = data[n_train:]

# helper to get a small batch (CPU->device)
def get_batch(data_tensor, block_size=block_size, batch_size=2, device=DEVICE):
    n = data_tensor.size(0)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# Greedy generation test
prompt = "System: You are SARAN. <|user|> Hello, how are you? <|assistant|> "
enc = tokenizer.encode(prompt, add_special_tokens=False)
idx = torch.tensor([enc + [eos]], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=32, temperature=0.0, top_k=None)
gen_txt = tokenizer.decode(out[0].tolist()[idx.shape[1]:], skip_special_tokens=True)
print("\nGREEDY SAMPLE:\n", gen_txt)

# Top-k probabilities for last token
model.eval()
with torch.no_grad():
    logits = model(idx[:, -model.block_size:], use_last_token=True)  # (1, V)
    probs = torch.softmax(logits, dim=-1)
    vals, ids_top = torch.topk(probs, k=20)
    print("\nTOP-K (token, prob):")
    for tokid, val in zip(ids_top[0].tolist(), vals[0].tolist()):
        try:
            t = tokenizer.decode([tokid])
        except Exception:
            t = str(tokid)
        print(f"{t!r}: {val:.6f}")

# Embedding gradient check (one backward step)
model.train()
xb, yb = get_batch(train_data, block_size=block_size, batch_size=2, device=DEVICE)
logits = model(xb, use_last_token=False)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
loss.backward()
tok_grad = model.tok_emb.weight.grad
print("\nloss:", float(loss.detach().cpu()))
print("tok_emb grad norm:", float(tok_grad.norm().cpu()) if tok_grad is not None else "None")

print("\nDiagnostics complete.")