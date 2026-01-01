#!/usr/bin/env python3
"""
ChatSARAN-Train.py

STRICT SARAN IMPLEMENTATION
Fully annotated to match Figure 1: The SARAN Architecture

This file preserves the exact architecture and training logic.
Only explanatory comments have been added.
"""

import math
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import trange

# =============================================================================
# ENV HELPERS
# =============================================================================

# Utility helpers to allow environment-variable overrides
def env_float(key, default):
    v = os.getenv(key)
    return float(v) if v is not None else default

def env_int(key, default):
    v = os.getenv(key)
    return int(v) if v is not None else default

def env_bool(key, default):
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Total number of tokens to train on (token-based training, not epoch-based)
MAX_TOKENS = env_int("MAX_TOKENS", 10_000_000)

# Context window size (sequence length)
BLOCK_SIZE = env_int("BLOCK_SIZE", 256)

# SARAN hidden dimension (d_model)
D_MODEL = env_int("D_MODEL", 384)

# Batch size per step
BATCH_SIZE = env_int("BATCH_SIZE", 32)

# Gradient accumulation for effective batch size
GRAD_ACCUM_STEPS = env_int("GRAD_ACCUM_STEPS", 4)

# Steps per epoch (derived from token budget)
DEFAULT_STEPS = max(1, int(MAX_TOKENS // (BATCH_SIZE * BLOCK_SIZE)))
STEPS_PER_EPOCH = env_int("STEPS_PER_EPOCH", DEFAULT_STEPS)

# Number of training epochs
EPOCHS = env_int("EPOCHS", 8)

# Learning rate (carefully chosen for no-LN SARAN)
LR = env_float("LR", 3e-4)

# Residual injection strength (critical for SARAN stability)
RESIDUAL_SCALE = env_float("RESIDUAL_SCALE", 0.2)

# Validation split
VAL_SPLIT = env_float("VAL_SPLIT", 0.05)

# Early stopping guard
EARLY_STOPPING_PATIENCE = env_int("EARLY_STOPPING_PATIENCE", 1000)

# Optimizer parameters
WEIGHT_DECAY = 0.0
BETA1, BETA2 = 0.9, 0.95

# Sampling hyperparameters (inference only)
TEMPERATURE = env_float("TEMPERATURE", 0.7)
TOP_K = env_int("TOP_K", 50)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 128)

# Disable compile/AMP for numerical stability
USE_TORCH_COMPILE = False
USE_AMP = False

# File paths
DATA_IDS = "data_ids.pt"
CKPT_PATH = "chat_saran_ckpt.pt"
BEST_CKPT_PATH = "chat_saran_ckpt.best.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"

# =============================================================================
# DEVICE SELECTION
# =============================================================================

# Automatically select CUDA, MPS, or CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", DEVICE)

# =============================================================================
# DATA LOADING & TOKENIZATION
# =============================================================================

# If tokenized data does not exist, rebuild it
rebuild = not os.path.isfile(DATA_IDS)

if rebuild:
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**6

    # Add padding and chat special tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})

    # Save tokenizer locally
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id

    # Data sources (streamed to avoid RAM blowup)
    sources = [
        ("openwebtext", "train"),
        ("wikitext", "wikitext-103-raw-v1"),
        ("daily_dialog", "train"),
        ("empathetic_dialogues", "train"),
    ]

    # Stream, tokenize, and chunk text into BLOCK_SIZE segments
    for name, split in sources:
        ds = load_dataset(name, split=split, streaming=True)
        for ex in ds:
            text = ex.get("text") or ex.get("dialog") or ""
            if isinstance(text, list):
                text = " ".join(text)
            text = text[:2000]
            if not text:
                continue

            toks = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(toks), BLOCK_SIZE):
                ids.extend(toks[i:i+BLOCK_SIZE] + [eos])
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

# Convert token list to tensor
data = torch.tensor(ids, dtype=torch.long)
vocab_size = tokenizer.vocab_size

# Train / validation split
n_train = int(len(data) * (1 - VAL_SPLIT))
train_data = data[:n_train]
val_data = data[n_train:]

# =============================================================================
# SARAN MODEL DEFINITION
# =============================================================================

class ChatSARAN(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, residual_scale):
        super().__init__()

        self.block_size = block_size
        self.residual_scale = residual_scale

        # (2) Token Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # (3) Positional Embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)

        # (5–7) Query, Key, Value projections (single head)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # Attention scaling factor
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model)))

        # (9) Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

        # (13) Output projection
        self.Wout = nn.Linear(d_model, vocab_size)

    def forward(self, idx, use_last_token=False):
        B, T = idx.shape

        # (1) Input Tokens → indices
        # (2) Token embeddings lookup
        tok = self.tok_emb(idx)

        # (3) Positional embeddings
        pos = 0.5 * self.pos_emb(torch.arange(T, device=idx.device))

        # (4) Embedding summation
        x = tok + pos

        # (5–7) Q, K, V projections
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # (8) Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # (9) Causal masking
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))

        # (10) Softmax normalization
        attn = F.softmax(scores, dim=-1)

        # (11) Attention output
        attn_out = attn @ v

        # (11.5) Residual injection (single residual path)
        x = x + self.residual_scale * attn_out

        # (12) Last-token selection
        last = x[:, -1, :]

        # (13–15) Output projection + implicit softmax in loss
        return self.Wout(last if use_last_token else x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:], use_last_token=True)

            # Sampling temperature
            logits = logits / TEMPERATURE

            # Top-k truncation
            if TOP_K:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -1e10

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)

        return idx

# =============================================================================
# TRAINING LOOP
# =============================================================================

def get_batch(data):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = ChatSARAN(vocab_size, D_MODEL, BLOCK_SIZE, RESIDUAL_SCALE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(BETA1, BETA2))

total_steps = EPOCHS * STEPS_PER_EPOCH
warmup_steps = max(1, int(0.01 * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

best_val = float("inf")
no_improve = 0

print("Beginning training...")
for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    optimizer.zero_grad()

    for step in trange(STEPS_PER_EPOCH, desc=f"Epoch {ep}/{EPOCHS}"):
        xb, yb = get_batch(train_data)
        logits = model(xb)

        # (15) Cross-entropy loss applies softmax implicitly
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        (loss / GRAD_ACCUM_STEPS).backward()
        running += loss.item()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(val_data)
        vloss = F.cross_entropy(
            model(xb).view(-1, vocab_size),
            yb.view(-1)
        ).item()

    print(f"Epoch {ep}: TrainLoss={running/STEPS_PER_EPOCH:.4f} ValLoss={vloss:.4f}")

    # Checkpointing
    if vloss < best_val:
        best_val = vloss
        torch.save(model.state_dict(), BEST_CKPT_PATH)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            break

    # Sample generation
    prompt = tokenizer.encode("Hello, how are you?", return_tensors="pt").to(DEVICE)
    out = model.generate(prompt, MAX_NEW_TOKENS)
    print("Sample:", tokenizer.decode(out[0], skip_special_tokens=True))

torch.save(model.state_dict(), CKPT_PATH)
print("Training complete.")
