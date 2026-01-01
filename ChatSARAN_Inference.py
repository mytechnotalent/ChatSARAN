#!/usr/bin/env python3
"""
ChatSARAN-Inference.py
MATCHES UPDATED ChatSARAN-Train.py EXACTLY
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# --------------------------- Hyperparams ---------------------------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_K = int(os.getenv("TOP_K", 50))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 128))

CKPT_PATH = os.getenv("CKPT_PATH", "chat_saran_ckpt.pt")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "chat_saran_tokenizer")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

# --------------------------- Load tokenizer ------------------------------------
if not os.path.isfile(CKPT_PATH):
    sys.exit(f"Checkpoint not found: {CKPT_PATH}")
if not os.path.isdir(TOKENIZER_DIR):
    sys.exit(f"Tokenizer dir not found: {TOKENIZER_DIR}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
tokenizer.model_max_length = 10**6
eos = tokenizer.eos_token_id

# --------------------------- Model ---------------------------------------------
class ChatSARAN(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, residual_scale):
        super().__init__()
        self.block_size = block_size
        self.residual_scale = residual_scale

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model)))
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

        self.Wout = nn.Linear(d_model, vocab_size)

    def forward(self, idx, use_last_token=False):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = 0.5 * self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn_out = attn @ v

        x = x + self.residual_scale * attn_out
        last = x[:, -1, :]

        return self.Wout(last if use_last_token else x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:], use_last_token=True)
            logits = logits / TEMPERATURE
            if TOP_K:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -1e10
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

# --------------------------- Load checkpoint -----------------------------------
state = torch.load(CKPT_PATH, map_location=DEVICE)

# NEW TRAINER SAVES STATE_DICT ONLY
if isinstance(state, dict) and "state_dict" not in state:
    # infer dimensions
    vocab_size = tokenizer.vocab_size
    d_model = state["tok_emb.weight"].shape[1]
    block_size = state["pos_emb.weight"].shape[0]
    residual_scale = 0.2
    model = ChatSARAN(vocab_size, d_model, block_size, residual_scale).to(DEVICE)
    model.load_state_dict(state)
else:
    model = ChatSARAN(
        state["vocab_size"],
        state["d_model"],
        state["block_size"],
        state.get("residual_scale", 0.2)
    ).to(DEVICE)
    model.load_state_dict(state["state_dict"])

model.eval()
print("ChatSARAN loaded. Ready.")

# --------------------------- Interactive loop ----------------------------------
SYSTEM_PROMPT = "System: You are SARAN. "

def clean(txt):
    return " ".join(txt.split()).strip()

try:
    while True:
        prompt = input("\nYou: ").strip()
        if not prompt:
            continue

        text = SYSTEM_PROMPT + "<|user|> " + prompt + " <|assistant|> "
        enc = tokenizer.encode(text, add_special_tokens=False)
        idx = torch.tensor([enc + [eos]], device=DEVICE)
        idx = idx[:, -model.block_size:]

        out = model.generate(idx, MAX_NEW_TOKENS)
        gen = out[0].tolist()[idx.shape[1]:]
        if eos in gen:
            gen = gen[:gen.index(eos)]
        print("SARAN:", clean(tokenizer.decode(gen, skip_special_tokens=True)))

except KeyboardInterrupt:
    print("\nExiting.")
