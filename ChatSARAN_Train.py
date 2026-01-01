#!/usr/bin/env python3
"""
ChatSARAN_Train.py
==================

STRICT SARAN + MINIMAL MLP (ARCHITECTURE PRESERVED)

This implementation follows the SARAN paper EXACTLY with a single,
non-stacked attention pass and a minimal MLP enhancement.

========================
THE 15 SARAN STEPS
========================
1.  Input Tokens (integer IDs)
2.  Token Embedding Lookup
3.  Positional Embedding Lookup
4.  Embedding Summation
5.  Query Projection
6.  Key Projection
7.  Value Projection
8.  Scaled Dot-Product Attention Scores
9.  Causal Masking (autoregressive)
10. Softmax → Attention Weights
11. Attention Output (Attn × V)
12. MLP Feature Synthesis (minimal, single)
13. Last Token Selection
14. Output Projection (+ bias)
15. Softmax (used implicitly by cross-entropy)

NO recurrence
NO deep stacking
NO multi-head attention
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


# ---------------- CONFIG ------------------------------------------------------
MAX_TOKENS = 10_000_000
BLOCK_SIZE = 768
D_MODEL = 768
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4
EPOCHS = 10_000
LR = 3e-4
RESIDUAL_SCALE = 0.2
VAL_SPLIT = 0.05
SEED = 42

DATA_IDS = "data_ids.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"
BEST_CKPT = "chat_saran_mlp.best.pt"
FINAL_CKPT = "chat_saran_mlp.pt"


# ---------------- DEVICE ------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", DEVICE)


# ---------------- SEEDING -----------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------- DATA --------------------------------------------------------
rebuild = not os.path.isfile(DATA_IDS)

if rebuild:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**9
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id

    sources = [
        ("openwebtext", None, "train"),
        ("wikitext", "wikitext-103-raw-v1", "train"),
    ]

    for name, config, split in sources:
        if config is None:
            ds = load_dataset(name, split=split, streaming=True)
        else:
            ds = load_dataset(name, config, split=split, streaming=True)

        for ex in ds:
            text = ex.get("text", "")[:2000]
            if not text:
                continue

            toks = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(toks), BLOCK_SIZE):
                ids.extend(toks[i:i + BLOCK_SIZE] + [eos])
                if len(ids) >= MAX_TOKENS:
                    break
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


# ---------------- MODEL -------------------------------------------------------
class ChatSARAN_MLP(nn.Module):
    """
    STRICT SARAN + MINIMAL MLP
    """

    def __init__(self, vocab_size, d_model, block_size, residual_scale):
        super().__init__()

        self.block_size = block_size
        self.residual_scale = residual_scale

        # (2) Token Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # (3) Positional Embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)

        # (5–7) QKV Projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # (12) Minimal MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # (8) Scale
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model)))

        # (9) Causal Mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).bool()
        )

        # (14) Output Projection
        self.Wout = nn.Linear(d_model, vocab_size)

    def forward(self, idx, use_last_token=False):
        B, T = idx.shape

        # (1–4) Embedding + Position
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        # (5–7) QKV
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # (8) Attention Scores
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # (9) Mask
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))

        # (10) Softmax
        attn = F.softmax(scores, dim=-1)

        # (11) Attention Output
        attn_out = attn @ v
        x = x + self.residual_scale * attn_out

        # (12) MLP
        x = x + self.mlp(x)

        # (13) Last Token
        last = x[:, -1, :]

        # (14) Output Projection
        return self.Wout(last if use_last_token else x)


# ---------------- TRAINING ----------------------------------------------------
def get_batch(data):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


model = ChatSARAN_MLP(vocab_size, D_MODEL, BLOCK_SIZE, RESIDUAL_SCALE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

steps_per_epoch = len(train_data) // (BATCH_SIZE * BLOCK_SIZE)
total_steps = EPOCHS * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    int(0.01 * total_steps),
    total_steps,
)

best_val = float("inf")

print("Beginning training...")
for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    optimizer.zero_grad()

    for step in trange(steps_per_epoch):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        (loss / GRAD_ACCUM_STEPS).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running += loss.item()

    avg_train = running / steps_per_epoch

    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(val_data)
        val_loss = F.cross_entropy(
            model(xb).view(-1, vocab_size),
            yb.view(-1)
        ).item()

    print(f"Epoch {ep}: TrainLoss={avg_train:.4f} ValLoss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), BEST_CKPT)

torch.save(model.state_dict(), FINAL_CKPT)
print("Training complete.")
