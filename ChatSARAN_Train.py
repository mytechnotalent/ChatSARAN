#!/usr/bin/env python3
"""
ChatSARAN_MLP_Train_v2.py

SARAN + Minimal MLP Extension (STRICT)
=====================================

This file implements SARAN EXACTLY as defined with ONE nonlinear MLP.
No recurrence. No stacking. Single-head attention.

15-Step Flow (STRICT):
1.  Input Tokens
2.  Token Embeddings
3.  Positional Embeddings
4.  Embedding Summation
5.  Query Projection
6.  Key Projection
7.  Value Projection
8.  Attention Score Calculation
9.  Causal Masking
10. Softmax (Attention Weights)
11. Attention Output Calculation
12. MLP Feature Synthesis (GEGLU)
13. Last Token Selection
14. Output Projection (Weight Tied)
15. Softmax (Loss)
"""

import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import trange

# ---------------------- CONFIG ---------------------------------
MAX_TOKENS = 10_000_000
BLOCK_SIZE = 256
D_MODEL = 768
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4
EPOCHS = 10000
LR = 1.5e-4
VAL_SPLIT = 0.05
SEED = 42

DATA_IDS = "data_ids.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"
BEST_CKPT = "chat_saran_mlp_v2.best.pt"
FINAL_CKPT = "chat_saran_mlp_v2.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", DEVICE)

# ---------------------- SEED -----------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------- DATA -----------------------------------
rebuild = not os.path.isfile(DATA_IDS)

if rebuild:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id

    sources = [("openwebtext", "train"), ("wikitext", "wikitext-103-raw-v1")]

    for name, split in sources:
        ds = load_dataset(name, split=split, streaming=True)
        for ex in ds:
            text = ex.get("text", "")[:2000]
            if not text:
                continue
            toks = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(toks), BLOCK_SIZE):
                ids.extend(toks[i:i+BLOCK_SIZE] + [eos])
                if len(ids) >= MAX_TOKENS:
                    break
            if len(ids) >= MAX_TOKENS:
                break

    ids = ids[:MAX_TOKENS]
    torch.save(ids, DATA_IDS)
else:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    ids = torch.load(DATA_IDS)

data = torch.tensor(ids, dtype=torch.long)
vocab_size = tokenizer.vocab_size

n_train = int(len(data) * (1 - VAL_SPLIT))
train_data = data[:n_train]
val_data = data[n_train:]

# ---------------------- MODEL ----------------------------------
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class ChatSARAN_MLP(nn.Module):
    """
    SARAN + MLP (15 STEPS STRICT)
    """

    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.block_size = block_size

        # (2) Token Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # (3) Positional Embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)

        # (5–7) QKV
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # (12) MLP (GEGLU)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 8 * d_model),
            GEGLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.mlp_scale = 0.1

        # (8) Attention scale
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model)))

        # (9) Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).bool()
        )

        # (14) Output projection (weight tied)
        self.Wout = nn.Linear(d_model, vocab_size, bias=False)
        self.Wout.weight = self.tok_emb.weight

    def forward(self, idx, use_last_token=False):
        B, T = idx.shape

        # (1–4) Embed + position
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        # (5–7) QKV
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # (8) Attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # (9) Mask
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))

        # (10) Softmax
        attn = F.softmax(scores, dim=-1)

        # (11) Attention output
        x = x + attn @ v

        # (12) MLP synthesis
        x = x + self.mlp_scale * self.mlp(x)

        # (13) Last token
        out = x[:, -1, :] if use_last_token else x

        # (14–15) Projection → softmax (loss outside)
        return self.Wout(out)

# ---------------------- TRAINING --------------------------------
def get_batch(data):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = ChatSARAN_MLP(vocab_size, D_MODEL, BLOCK_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_batches = len(train_data) // (BATCH_SIZE * BLOCK_SIZE)
total_steps = EPOCHS * num_batches
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(0.01 * total_steps), total_steps
)

best_val = float("inf")
print("Beginning training...")

for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    optimizer.zero_grad()

    for step in trange(num_batches):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        (loss / GRAD_ACCUM_STEPS).backward()
        running += loss.item()

        if step % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    train_loss = running / num_batches

    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(val_data)
        val_loss = F.cross_entropy(
            model(xb).view(-1, vocab_size), yb.view(-1)
        ).item()

    print(f"Epoch {ep}: TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), BEST_CKPT)

torch.save(model.state_dict(), FINAL_CKPT)
print("Training complete.")
