#!/usr/bin/env python3
"""
ChatSARAN_Train.py
==================

SARAN: Shallow Auto-Regressive Attention Network
-------------------------------------------------
Faithful implementation of the 15-step SARAN architecture.

STRICT 15-STAGE COMPUTATIONAL GRAPH:
------------------------------------
1.  Input Tokens         - Integer token IDs [t1, t2, ..., tn]
2.  Token Embeddings     - Lookup via Wembed (trainable)
3.  Positional Encodings - Lookup via Wpos (trainable)
4.  Embedding Summation  - X = Embed(T) + Pos
5.  Query Projection     - Q = X @ Wq
6.  Key Projection       - K = X @ Wk
7.  Value Projection     - V = X @ Wv
8.  Attention Scores     - Scores = (Q @ K^T) / sqrt(d_model)
9.  Causal Masking       - Apply -inf mask to future positions
10. Softmax              - Attention weights = softmax(Scores)
11. Attention Output     - Attn_out = Weights @ V
12. Residual Connection  - H = X + Attn_out
13. Output Projection    - Logits = H @ Wout
14. Bias Addition        - Logits = Logits + bout
15. Softmax (implicit)   - Cross-entropy loss applies softmax

MINIMAL EXTENSIONS FOR TRAINING STABILITY:
------------------------------------------
E1. LayerNorm after embedding (pre-attention normalization)
E2. LayerNorm before output projection (post-attention normalization)
E3. Gradient clipping (prevents explosion)

NOTE: We predict ALL positions (not just last token) for efficient training.
"""

import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


# =========================
# CONFIG
# =========================
MAX_TOKENS = 10_000_000  # Total tokens to use
BLOCK_SIZE = 256  # Context window (smaller = faster iteration)
D_MODEL = 512  # Embedding dimension
N_HEADS = 8  # Multi-head attention for better learning
BATCH_SIZE = 64  # Batch size
GRAD_ACCUM_STEPS = 4  # Gradient accumulation
EPOCHS = 100  # Training epochs
LR = 1e-3  # Learning rate
MIN_LR = 1e-5  # Minimum learning rate
WARMUP_STEPS = 500  # Warmup steps
GRAD_CLIP = 1.0  # Gradient clipping threshold
DROPOUT = 0.1  # Dropout for regularization
VAL_SPLIT = 0.05  # Validation split
SEED = 42
EVAL_INTERVAL = 1  # Evaluate every N epochs
PATIENCE = 10  # Early stopping patience

DATA_IDS = "data_ids.pt"
TOKENIZER_DIR = "saran_tokenizer"
BEST_CKPT = "saran.best.pt"
FINAL_CKPT = "saran.final.pt"


# =========================
# DEVICE
# =========================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Device: {DEVICE}")


# =========================
# SEEDING
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DATA LOADING
# =========================
rebuild = not os.path.isfile(DATA_IDS)

if rebuild:
    print("Building dataset from scratch...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.model_max_length = 10**9
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id

    sources = [
        ("openwebtext", None, "train"),
        ("wikitext", "wikitext-103-raw-v1", "train"),
    ]

    for name, cfg, split in sources:
        print(f"Loading {name}...")
        ds = load_dataset(name, cfg, split=split, streaming=True)
        for ex in ds:
            text = ex.get("text", "")[:4000]
            if not text or len(text.strip()) < 50:
                continue
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.extend(toks + [eos])
            if len(ids) >= MAX_TOKENS:
                break
        if len(ids) >= MAX_TOKENS:
            break

    ids = ids[:MAX_TOKENS]
    torch.save(ids, DATA_IDS)
    print(f"Saved {len(ids):,} tokens to {DATA_IDS}")
else:
    print(f"Loading cached data from {DATA_IDS}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    ids = torch.load(DATA_IDS)
    print(f"Loaded {len(ids):,} tokens")

data = torch.tensor(ids, dtype=torch.long)
VOCAB_SIZE = tokenizer.vocab_size
print(f"Vocabulary size: {VOCAB_SIZE:,}")

# Train/val split
n_train = int(len(data) * (1 - VAL_SPLIT))
train_data = data[:n_train]
val_data = data[n_train:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


# =========================
# MODEL: SARAN
# =========================
class SARAN(nn.Module):
    """
    Shallow Auto-Regressive Attention Network

    Single-block self-attention following the 15-step architecture.
    Uses multi-head attention for better representational capacity.
    """

    def __init__(self, vocab_size, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size

        # ===== STEP 2: Token Embeddings =====
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # ===== STEP 3: Positional Encodings =====
        self.pos_emb = nn.Embedding(block_size, d_model)

        # ===== EXTENSION E1: Pre-attention LayerNorm =====
        self.ln1 = nn.LayerNorm(d_model)

        # ===== STEPS 5-7: Q, K, V Projections =====
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # Output projection for multi-head attention
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # ===== EXTENSION E2: Post-attention LayerNorm =====
        self.ln2 = nn.LayerNorm(d_model)

        # ===== STEPS 13-14: Output Projection + Bias =====
        self.Wout = nn.Linear(d_model, vocab_size)  # includes bias (Step 14)

        # Dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ===== STEP 8: Scaling factor =====
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # ===== STEP 9: Causal mask =====
        self.register_buffer(
            "mask", torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        Forward pass through the 15-step SARAN architecture.

        Args:
            idx: Input token IDs, shape (B, T)

        Returns:
            logits: Output logits, shape (B, T, vocab_size)
        """
        B, T = idx.shape

        # ===== STEP 1: Input Tokens =====
        # idx contains integer token IDs

        # ===== STEP 2: Token Embeddings =====
        tok = self.tok_emb(idx)  # (B, T, d_model)

        # ===== STEP 3: Positional Encodings =====
        positions = torch.arange(T, device=idx.device)
        pos = self.pos_emb(positions)  # (T, d_model)

        # ===== STEP 4: Embedding Summation =====
        x = tok + pos  # (B, T, d_model)

        # ===== EXTENSION E1: Pre-attention LayerNorm =====
        x_norm = self.ln1(x)

        # ===== STEP 5: Query Projection =====
        Q = self.Wq(x_norm)  # (B, T, d_model)

        # ===== STEP 6: Key Projection =====
        K = self.Wk(x_norm)  # (B, T, d_model)

        # ===== STEP 7: Value Projection =====
        V = self.Wv(x_norm)  # (B, T, d_model)

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # ===== STEP 8: Attention Score Calculation =====
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # ===== STEP 9: Causal Masking =====
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))

        # ===== STEP 10: Softmax =====
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # ===== STEP 11: Attention Output =====
        attn_out = attn_weights @ V  # (B, H, T, D)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_out = self.Wo(attn_out)  # Project back
        attn_out = self.resid_dropout(attn_out)

        # ===== STEP 12: Residual Connection =====
        h = x + attn_out  # (B, T, d_model)

        # ===== EXTENSION E2: Post-attention LayerNorm =====
        h = self.ln2(h)

        # ===== STEPS 13-14: Output Projection + Bias =====
        logits = self.Wout(h)  # (B, T, vocab_size)

        # ===== STEP 15: Softmax (implicit in cross-entropy loss) =====
        return logits


# =========================
# TRAINING UTILITIES
# =========================
def get_batch(data, block_size, batch_size, device):
    """Sample a random batch of sequences."""
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model, train_data, val_data, block_size, batch_size, device, eval_iters=50
):
    """Estimate loss on train and val sets."""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
        losses[name] = total_loss / eval_iters
    model.train()
    return losses


def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =========================
# MODEL INITIALIZATION
# =========================
print("\nInitializing SARAN model...")
model = SARAN(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1
)

# Calculate steps
steps_per_epoch = max(
    1, len(train_data) // (BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS)
)
total_steps = EPOCHS * steps_per_epoch
print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps:,}")


# =========================
# TRAINING LOOP
# =========================
print("\n" + "=" * 60)
print("BEGINNING TRAINING")
print("=" * 60 + "\n")

best_val_loss = float("inf")
patience_counter = 0
global_step = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:3d}")

    for step in pbar:
        # Gradient accumulation
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)

            logits = model(x)
            # Predict ALL tokens, not just the last one
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item()

        # Update learning rate
        lr = get_lr(global_step, WARMUP_STEPS, LR, MIN_LR, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient clipping (EXTENSION E3)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        pbar.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}", "lr": f"{lr:.2e}"})

    avg_train_loss = epoch_loss / steps_per_epoch

    # Evaluation
    if epoch % EVAL_INTERVAL == 0:
        losses = estimate_loss(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE
        )
        val_loss = losses["val"]

        print(
            f"Epoch {epoch:3d} | Train: {losses['train']:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                BEST_CKPT,
            )
            print(f"  → New best! Saved to {BEST_CKPT}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"\nEarly stopping after {epoch} epochs (no improvement for {PATIENCE} epochs)"
                )
                break

        # Target achieved?
        if val_loss < 2.0:
            print(f"\n🎉 TARGET ACHIEVED! Val loss {val_loss:.4f} < 2.0")
            break

# Save final model
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val_loss,
    },
    FINAL_CKPT,
)

print("\n" + "=" * 60)
print(f"TRAINING COMPLETE")
print(f"Best validation loss: {best_val_loss:.4f}")
print("=" * 60)
