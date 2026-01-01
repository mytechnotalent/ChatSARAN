#!/usr/bin/env python3
"""
ChatSARAN_MLP_Inference_v2.py

STRICT SARAN Inference (15 Steps)
================================

Uses the trained SARAN+MLP model exactly as trained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

BLOCK_SIZE = 256
D_MODEL = 768
TEMPERATURE = 0.7
TOP_K = 50
MAX_NEW_TOKENS = 200

MODEL_CKPT = "chat_saran_mlp_v2.best.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", DEVICE)

# ---------------------- MODEL ----------------------------------
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class ChatSARAN_MLP(nn.Module):
    """
    SARAN + MLP (15 STEP INFERENCE FLOW)
    """

    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 8 * d_model),
            GEGLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.mlp_scale = 0.1

        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model)))
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).bool()
        )

        self.Wout = nn.Linear(d_model, vocab_size, bias=False)
        self.Wout.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)

        x = x + attn @ v
        x = x + self.mlp_scale * self.mlp(x)

        return self.Wout(x[:, -1, :])

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])
            logits /= TEMPERATURE

            if TOP_K:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, [-1]]] = -1e10

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ---------------------- LOAD -----------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = ChatSARAN_MLP(tokenizer.vocab_size, D_MODEL, BLOCK_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
model.eval()

# ---------------------- RUN ------------------------------------
prompt = "Hello, how are you?"
ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
out = model.generate(ids, MAX_NEW_TOKENS)
print(tokenizer.decode(out[0], skip_special_tokens=True))
