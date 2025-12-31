#!/usr/bin/env python3

"""
ChatSARAN-Inference.py — complete inference script matching Train

Features:
 - Loads tokenizer saved by Train (chat_saran_tokenizer)
 - Loads checkpoint saved by Train (chat_saran_ckpt.pt)
 - Reconstructs strict single-block SARAN (explicit Steps 1..15).
 - Residual scaling honored from checkpoint (residual_scale in saved dict).
 - Simple interactive loop for chatting (greedy or sampled decoding).
Usage:
  python ChatSARAN-Inference.py
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# --------------------------- Hyperparams (edit if desired) ---------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.20))
TOP_K = int(os.getenv("TOP_K", 40))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 128))
CKPT_PATH = os.getenv("CKPT_PATH", "chat_saran_ckpt.pt")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "chat_saran_tokenizer")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
# ------------------------------------------------------------------------------

if not os.path.isfile(CKPT_PATH):
    sys.exit(f"Checkpoint not found: {CKPT_PATH}")
if not os.path.isdir(TOKENIZER_DIR):
    sys.exit(f"Tokenizer directory not found: {TOKENIZER_DIR}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
# mirror train behavior: allow long encodes (we chunk before feeding model)
tokenizer.model_max_length = 10**6
eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

# --------------------------- Model matching Train --------------------------------
class ChatSARAN(nn.Module):
    def __init__(self, vocab_size, d_model=384, block_size=256, residual_scale=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.residual_scale = float(residual_scale)

        # Step 1: token embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Step 2: positional embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)
        # Steps 5-7: Q/K/V
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        # Step 8: scaling constant
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model), dtype=torch.float32))
        # Step 9: causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
        # Steps 13-14: output projection
        self.Wout = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, idx: torch.LongTensor, use_last_token: bool = False):
        B, T = idx.shape
        assert T <= self.block_size, f"seq len {T} > block_size {self.block_size}"

        # Steps 1-4
        tok = self.tok_emb(idx)                                    # (B, T, d)
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))  # (1, T, d)
        x = tok + pos                                               # (B, T, d)

        # Steps 5-7
        q = self.Wq(x); k = self.Wk(x); v = self.Wv(x)

        # Step 8
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Step 9
        m = self.mask[:T, :T].to(scores.device)
        scores = scores.masked_fill(~m, float("-inf"))

        # Step 10
        attn = F.softmax(scores, dim=-1)

        # Step 11
        attn_out = attn @ v

        # Step 11.5: scaled residual
        x = x + self.residual_scale * attn_out

        # Step 12
        last = x[:, -1, :]

        # Steps 13-15
        if use_last_token:
            return self.Wout(last)
        return self.Wout(x)

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int = 64, temperature: float = None, top_k: int = None):
        temperature = TEMPERATURE if temperature is None else float(temperature)
        top_k = TOP_K if top_k is None else top_k

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:].to(next(self.parameters()).device)
            logits = self(idx_cond, use_last_token=True)  # (B, V)

            if temperature is not None and float(temperature) <= 0.0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits_proc = logits / (temperature if temperature is not None else 1.0)
                if top_k is not None:
                    k = min(top_k, logits_proc.size(-1))
                    v, _ = torch.topk(logits_proc, k)
                    cutoff = v[:, -1].unsqueeze(1)
                    logits_proc = torch.where(logits_proc < cutoff, torch.full_like(logits_proc, float("-1e10")), logits_proc)
                probs = F.softmax(logits_proc, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx.to(next(self.parameters()).device), nxt], dim=1)
        return idx

# --------------------------- Load checkpoint & run interactive loop ----------------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
if tokenizer.vocab_size != ckpt.get("vocab_size"):
    sys.exit(f"Tokenizer vocab_size ({tokenizer.vocab_size}) != checkpoint vocab_size ({ckpt.get('vocab_size')}). "
             "Use the tokenizer saved during training (chat_saran_tokenizer/).")

model = ChatSARAN(ckpt["vocab_size"], ckpt["d_model"], ckpt["block_size"], residual_scale=ckpt.get("residual_scale", 0.1)).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Loaded ChatSARAN model. Ready to chat. (Ctrl-C to exit)")

def clean_decode(token_ids):
    txt = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return " ".join(txt.split()).strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "System: You are SARAN, a helpful, concise assistant. ")

try:
    while True:
        prompt = input("\nYou: ").strip()
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        user_block = "<|user|> " + prompt + " <|assistant|> "
        full = SYSTEM_PROMPT + user_block
        enc = tokenizer.encode(full, add_special_tokens=False)
        idx = torch.tensor([enc + [eos]], dtype=torch.long, device=DEVICE)
        if idx.shape[1] > model.block_size:
            idx = idx[:, -model.block_size:]

        out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
        gen_ids = out[0].tolist()[idx.shape[1]:]
        if eos in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos)]
        reply = clean_decode(gen_ids)
        print("SARAN:", reply)
except KeyboardInterrupt:
    print("\nExiting.")