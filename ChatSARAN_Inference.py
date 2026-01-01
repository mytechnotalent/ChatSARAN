#!/usr/bin/env python3
"""
ChatSARAN_MLP_Inference.py

Inference-only script for SARAN + Minimal MLP Extension.

This file EXACTLY mirrors the training architecture and explicitly documents
the 15 SARAN computational steps in forward order.

No recurrence. No stacking. Single-head attention.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


# --------------------------- CONFIG -------------------------------------------
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", 256))
D_MODEL = int(os.getenv("D_MODEL", 768))

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_K = int(os.getenv("TOP_K", 50))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 200))

CKPT_PATH = os.getenv("CKPT_PATH", "chat_saran_mlp.best.pt")
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


# --------------------------- TOKENIZER ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
vocab_size = tokenizer.vocab_size


# --------------------------- MODEL --------------------------------------------
class ChatSARAN_MLP(nn.Module):
    """
    === SARAN + MLP (INFERENCE) ===

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
    15. Softmax (Sampling)
    """

    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()

        self.block_size = block_size

        # (2) Token Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # (3) Positional Embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)

        # (5–7) QKV Projections (single-head)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # (12) Minimal MLP Extension
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # (8) Scaling factor
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model)))

        # (9) Causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).bool()
        )

        # (14) Output projection
        self.Wout = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        """
        Forward pass used for NEXT TOKEN prediction.
        """
        B, T = idx.shape

        # (1–4) Token + Positional Embedding Summation
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        # (5–7) Q, K, V projections
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # (8) Scaled Dot-Product Attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # (9) Causal masking (autoregressive)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))

        # (10) Softmax → attention weights
        attn = F.softmax(scores, dim=-1)

        # (11) Attention output
        attn_out = attn @ v
        x = x + attn_out

        # (12) MLP feature synthesis
        x = x + self.mlp(x)

        # (13) Select last token hidden state
        last = x[:, -1, :]

        # (14) Output projection to vocab logits
        return self.Wout(last)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation loop.
        """
        for _ in range(max_new_tokens):

            # Crop context if needed
            idx_cond = idx[:, -self.block_size :]

            # Forward pass
            logits = self(idx_cond)

            # (15) Sampling softmax
            logits /= TEMPERATURE

            if TOP_K > 0:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -1e10

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx


# --------------------------- LOAD MODEL ---------------------------------------
model = ChatSARAN_MLP(vocab_size, D_MODEL, BLOCK_SIZE).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

print("Model loaded.")


# --------------------------- INTERACTIVE LOOP ---------------------------------
print("\n=== SARAN-MLP Interactive Inference ===")
print("Type a prompt and press Enter. Ctrl+C to exit.\n")

while True:
    try:
        prompt = input("Prompt: ").strip()
        if not prompt:
            continue

        tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(tokens, MAX_NEW_TOKENS)

        print("\n--- Output ---")
        print(tokenizer.decode(out[0], skip_special_tokens=True))
        print("\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
