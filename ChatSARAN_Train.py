#!/usr/bin/env python3
"""
ChatSARAN_MLP_Train.py

SARAN + Minimal MLP Extension
============================

This file implements SARAN EXACTLY as defined (15 steps),
with ONE minimal MLP added for nonlinear feature synthesis.

The computational flow is strictly left-to-right.
No recurrence. No deep stacking. Single-head attention only.
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


# -------------------------- ENV HELPERS ---------------------------------------
def env_int(key, default):
    return int(os.getenv(key, default))


def env_float(key, default):
    return float(os.getenv(key, default))


# --------------------------- HYPERPARAMETERS ----------------------------------
MAX_TOKENS = env_int("MAX_TOKENS", 10_000_000)
BLOCK_SIZE = env_int("BLOCK_SIZE", 256)

D_MODEL = env_int("D_MODEL", 768)
BATCH_SIZE = env_int("BATCH_SIZE", 32)
GRAD_ACCUM_STEPS = env_int("GRAD_ACCUM_STEPS", 4)

EPOCHS = env_int("EPOCHS", 10000)
LR = env_float("LR", 3e-4)
RESIDUAL_SCALE = env_float("RESIDUAL_SCALE", 0.2)

VAL_SPLIT = env_float("VAL_SPLIT", 0.05)
SEED = env_int("SEED", 42)

TEMPERATURE = env_float("TEMPERATURE", 0.7)
TOP_K = env_int("TOP_K", 50)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 128)

DATA_IDS = "data_ids.pt"
BEST_CKPT = "chat_saran_mlp.best.pt"
FINAL_CKPT = "chat_saran_mlp.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"


# --------------------------- DEVICE -------------------------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", DEVICE)


# --------------------------- SEEDING ------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------- DATA ---------------------------------------------
rebuild = not os.path.isfile(DATA_IDS)

if rebuild:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**6
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id

    sources = [
        ("openwebtext", "train"),
        ("wikitext", "wikitext-103-raw-v1"),
    ]

    for name, split in sources:
        ds = load_dataset(name, split=split, streaming=True)
        for ex in ds:
            text = ex.get("text", "")[:2000]
            if not text:
                continue
            toks = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(toks), BLOCK_SIZE):
                ids.extend(toks[i : i + BLOCK_SIZE] + [eos])
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


# --------------------------- MODEL --------------------------------------------
class ChatSARAN_MLP(nn.Module):
    """
    === SARAN + MLP (FULL 15 STEP FLOW) ===

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
    12. MLP Feature Synthesis (Extension)
    13. Last Token Selection
    14. Output Projection (+ Bias)
    15. Softmax (loss / sampling)
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

        # (12) Minimal MLP Extension
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

        # (1–4) Embed + Position
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

        # (12) MLP Feature Synthesis
        x = x + self.mlp(x)

        # (13) Last Token
        last = x[:, -1, :]

        # (14) Output Projection
        return self.Wout(last if use_last_token else x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size :], use_last_token=True)
            logits /= TEMPERATURE

            if TOP_K:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -1e10

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# --------------------------- TRAINING ------------------------------------------
def get_batch(data):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


model = ChatSARAN_MLP(vocab_size, D_MODEL, BLOCK_SIZE, RESIDUAL_SCALE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

total_steps = EPOCHS * (len(train_data) // (BATCH_SIZE * BLOCK_SIZE))
warmup_steps = max(1, int(0.01 * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

best_val = float("inf")

print("Beginning training...")
for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    optimizer.zero_grad()

    for _ in trange(len(train_data) // (BATCH_SIZE * BLOCK_SIZE)):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        (loss / GRAD_ACCUM_STEPS).backward()
        running += loss.item()

        if _ % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(val_data)
        vloss = F.cross_entropy(
            model(xb).view(-1, vocab_size), yb.view(-1)
        ).item()

    print(f"Epoch {ep}: TrainLoss={running:.4f} ValLoss={vloss:.4f}")

    if vloss < best_val:
        best_val = vloss
        torch.save(model.state_dict(), BEST_CKPT)

    prompt = tokenizer.encode("Hello, how are you?", return_tensors="pt").to(DEVICE)
    out = model.generate(prompt, MAX_NEW_TOKENS)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

torch.save(model.state_dict(), FINAL_CKPT)
print("Training complete.")
