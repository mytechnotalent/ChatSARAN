#!/usr/bin/env python3

"""
ChatSARAN-Inference.py
alpha 0.0.2

Inference script matching the training script variant with residual scaling.
Residual scaling applied in the forward (x = x + 0.1 * attn_out).
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# --------------------------- HYPERPARAMS (EDIT AT TOP) ---------------------------
TEMPERATURE = 0.20   # set to 0.0 for greedy deterministic decoding
TOP_K = 40
MAX_NEW_TOKENS = 128
# --------------------------------------------------------------------------

CKPT_PATH = "chat_saran_ckpt.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")

if not os.path.isfile(CKPT_PATH):
    sys.exit(f"Checkpoint not found: {CKPT_PATH}")
if not os.path.isdir(TOKENIZER_DIR):
    sys.exit(f"Tokenizer directory not found: {TOKENIZER_DIR}")

# load tokenizer (local only) and mirror train's model_max_length setting
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
tokenizer.model_max_length = 10**6  # mirror train to avoid warnings when encoding then chunking
eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

class ChatSARAN(nn.Module):
    """
    Inference ChatSARAN with residual scaling (no LayerNorm).
    Steps mapping remains consistent with the training file (1..15, residual at 11.5).
    """
    def __init__(self, vocab_size, d_model=384, block_size=256, residual_scale: float = 0.1):
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
        # Step 8: scaling constant (buffer)
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
        q = self.Wq(x); k = self.Wk(x); v = self.Wv(x)              # each (B, T, d)

        # Step 8
        scores = (q @ k.transpose(-2, -1)) * self.scale             # (B, T, T)
        # Step 9
        m = self.mask[:T, :T].to(scores.device)
        scores = scores.masked_fill(~m, float("-inf"))
        # Step 10
        attn = F.softmax(scores, dim=-1)                            # (B, T, T)
        # Step 11
        attn_out = attn @ v                                          # (B, T, d)

        # Step 11.5: residual scaling (no LayerNorm)
        x = x + self.residual_scale * attn_out

        # Steps 12-15
        last = x[:, -1, :]
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
                temp = float(temperature) if temperature is not None else 1.0
                logits = logits / temp
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    cutoff = v[:, -1].unsqueeze(1)
                    logits = torch.where(logits < cutoff, torch.full_like(logits, float("-1e10")), logits)
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx.to(next(self.parameters()).device), nxt], dim=1)
        return idx

# --------------------------- Load checkpoint and model ---------------------------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
if tokenizer.vocab_size != ckpt.get("vocab_size"):
    sys.exit(f"Tokenizer vocab_size ({tokenizer.vocab_size}) != checkpoint vocab_size ({ckpt.get('vocab_size')}). "
             "Please use the tokenizer saved during training (chat_saran_tokenizer/).")

rs = ckpt.get("residual_scale", 0.1)
model = ChatSARAN(ckpt["vocab_size"], ckpt["d_model"], ckpt["block_size"], residual_scale=rs).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Loaded ChatSARAN model. Ready to chat. (Ctrl-C to exit)")

def clean_decode(token_ids):
    txt = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    txt = " ".join(txt.split())
    return txt.strip()

SYSTEM_PROMPT = "System: You are SARAN, a helpful, concise, and polite assistant. "

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