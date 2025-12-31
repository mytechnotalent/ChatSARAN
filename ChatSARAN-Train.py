#!/usr/bin/env python3

"""
ChatSARAN-Train.py
alpha 0.0.2

Single-block SARAN training (no LayerNorm) with:
 - residual scaling: x = x + 0.1 * attn_out
 - warmup increased to 5% of total steps
 - weight_decay set to 0.0 during initial experiments
 - chunked tokenization, rebuild data_ids.pt if needed
 - conservative model/batch defaults for single-machine training
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

# --------------------------- HYPERPARAMS (EDIT AT TOP) ---------------------------
MAX_TOKENS = 10_000_000   # total tokens to build
BLOCK_SIZE = 256          # context length (T)
D_MODEL = 384             # model dimensionality (reduced for single-machine)
BATCH_SIZE = 32           # physical batch per step
GRAD_ACCUM_STEPS = 4      # accumulate 4 steps -> effective batch 128

# compute steps_per_epoch so an epoch ≈ MAX_TOKENS tokens seen
STEPS_PER_EPOCH = max(1, int(MAX_TOKENS // (BATCH_SIZE * BLOCK_SIZE)))
EPOCHS = 10_000           # long budget; early stopping used below
LR = 5e-5                 # learning rate
VAL_SPLIT = 0.05

# early stopping
EARLY_STOPPING_PATIENCE = 20   # stop if val doesn't improve for this many epochs

# validation / sampling frequency (reduce overhead)
VAL_STEPS_LIMIT = 50       # max val steps per epoch (keeps val cheap)
SAMPLE_EVERY = 5           # print a sample every N epochs (0 = disabled)

# optional compile (try it, fall back if unsupported)
USE_TORCH_COMPILE = True

# decoding (used only for sample printing at epoch end)
TEMPERATURE = 0.20
TOP_K = 40
MAX_NEW_TOKENS = 128
# -------------------------------------------------------------------------------

DATA_IDS = "data_ids.pt"
CKPT_PATH = "chat_saran_ckpt.pt"
BEST_CKPT_PATH = "chat_saran_ckpt.best.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
print("Device:", DEVICE)

# --------------------------- Prepare tokenizer + data (stream safe, chunking) ---------------------------
rebuild_data = False
if os.path.isfile(DATA_IDS):
    try:
        ids_existing = torch.load(DATA_IDS)
        print(f"Found {DATA_IDS} with {len(ids_existing)} tokens.")
        if len(ids_existing) < MAX_TOKENS:
            print("Existing token file smaller than MAX_TOKENS -> will rebuild.")
            rebuild_data = True
        else:
            ids = ids_existing[:MAX_TOKENS]
    except Exception:
        print("Failed to load existing data_ids.pt -> rebuilding.")
        rebuild_data = True
else:
    rebuild_data = True

if rebuild_data:
    print("Preparing tokenizer and streaming datasets to build token stream (chunking long pieces)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # allow long encodes without noisy warnings; we will chunk into BLOCK_SIZE pieces immediately
    tokenizer.model_max_length = 10**6
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

    # Datasets to stream (will stop when MAX_TOKENS reached)
    datasets_to_stream = [
        ("databricks/databricks-dolly-15k", "train"),
        ("openwebtext", "train"),
        ("wikitext", "wikitext-103-raw-v1"),
        ("daily_dialog", "train"),
        ("empathetic_dialogues", "train"),
    ]

    for ds_name, split in datasets_to_stream:
        try:
            print("Streaming:", ds_name, split)
            ds = load_dataset(ds_name, split=split, streaming=True)
            for ex in ds:
                # prioritize conversational fields where present
                inst = (ex.get("instruction") or ex.get("prompt") or "")[:2000]
                resp = (ex.get("response") or ex.get("completion") or "")[:2000]
                if inst or resp:
                    piece = "<|user|> " + inst + " <|assistant|> " + resp
                else:
                    text = (ex.get("text") or ex.get("article") or "")
                    if isinstance(text, list):
                        text = " ".join(text)
                    piece = text[:2000]

                if not piece:
                    continue

                # encode, then chunk into BLOCK_SIZE token blocks to avoid long sequence warnings
                tokens = tokenizer.encode(piece, add_special_tokens=False)
                if len(tokens) <= BLOCK_SIZE:
                    ids.extend(tokens + [eos])
                else:
                    # break into blocks of BLOCK_SIZE (keeps more content and avoids warnings)
                    for i in range(0, len(tokens), BLOCK_SIZE):
                        chunk = tokens[i:i + BLOCK_SIZE]
                        if len(chunk) == 0:
                            continue
                        ids.extend(chunk + [eos])
                        if len(ids) >= MAX_TOKENS:
                            break

                if len(ids) >= MAX_TOKENS:
                    break
        except Exception as e:
            # dataset may not be available in your environment; skip if it errors
            print(f"  skipping dataset {ds_name}: {e}")
        if len(ids) >= MAX_TOKENS:
            break

    ids = ids[:MAX_TOKENS]
    torch.save(ids, DATA_IDS)
    print("Saved processed token ids ->", DATA_IDS, " (tokens):", len(ids))
else:
    # ensure tokenizer is loaded if we didn't rebuild
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    # mirror training behavior: allow long encodes without warnings
    tokenizer.model_max_length = 10**6
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

# --------------------------- Prepare data tensors (keep on CPU until minibatch) ---------------------------
data = torch.tensor(ids, dtype=torch.long)   # keep on CPU, move minibatches to DEVICE later
vocab_size = tokenizer.vocab_size
print(f"Built token stream: {len(data)} tokens | vocab_size={vocab_size}")

# --------------------------- Train/Val split (keep on CPU) ---------------------------
n_train = int(len(data) * (1.0 - VAL_SPLIT))
train_data = data[:n_train]    # CPU tensor
val_data = data[n_train:]      # CPU tensor
print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

# --------------------------- ChatSARAN model (STRICT SARAN steps with residual scaling) ---------------------------
class ChatSARAN(nn.Module):
    """
    Single-block masked self-attention with a small residual scaling factor.
    Residual scaling applied: x = x + 0.1 * attn_out (step 11.5).
    LayerNorm intentionally removed to keep strict SARAN.
    """
    def __init__(self, vocab_size: int, d_model: int = D_MODEL, block_size: int = BLOCK_SIZE, residual_scale: float = 0.1):
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
        # Step 8: scaling constant buffer
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model), dtype=torch.float32))
        # Step 9: causal mask buffer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
        # Steps 13-14: output projection
        self.Wout = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, idx: torch.LongTensor, use_last_token: bool = False):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        # Steps 1-4
        tok = self.tok_emb(idx)                                      # (B, T, d)
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))  # (1, T, d)
        x = tok + pos                                                 # (B, T, d)

        # Steps 5-7
        q = self.Wq(x)                                                # (B, T, d)
        k = self.Wk(x)                                                # (B, T, d)
        v = self.Wv(x)                                                # (B, T, d)

        # Step 8
        scores = (q @ k.transpose(-2, -1)) * self.scale               # (B, T, T)
        # Step 9: causal mask
        m = self.mask[:T, :T].to(scores.device)
        scores = scores.masked_fill(~m, float("-inf"))
        # Step 10
        attn = F.softmax(scores, dim=-1)                              # (B, T, T)
        # Step 11
        attn_out = attn @ v                                           # (B, T, d)

        # Step 11.5: residual scaling (keeps SARAN ordering)
        x = x + self.residual_scale * attn_out

        # Step 12
        last = x[:, -1, :]                                            # (B, d)
        # Steps 13-15
        if use_last_token:
            return self.Wout(last)                                    # (B, V)
        return self.Wout(x)                                           # (B, T, V)

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
                temp = float(temperature)
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

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "block_size": self.block_size,
            "residual_scale": self.residual_scale,
        }, path)

    @classmethod
    def load(cls, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or DEVICE)
        rs = ckpt.get("residual_scale", 0.1)
        model = cls(ckpt["vocab_size"], ckpt["d_model"], ckpt["block_size"], residual_scale=rs)
        model.load_state_dict(ckpt["state_dict"])
        return model

# --------------------------- Batching helper (moves minibatch to DEVICE only) ---------------------------
def get_batch_from_concat(data_tensor: torch.LongTensor, block_size: int = BLOCK_SIZE, batch_size: int = BATCH_SIZE, device=DEVICE) -> Tuple[torch.LongTensor, torch.LongTensor]:
    n = data_tensor.size(0)
    if n <= block_size + 1:
        raise ValueError("Data tensor too small for block_size")
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --------------------------- Instantiate + optimizer + scheduler ---------------------------
model = ChatSARAN(vocab_size=vocab_size, d_model=D_MODEL, block_size=BLOCK_SIZE, residual_scale=0.1).to(DEVICE)

# try torch.compile to speed up if available
if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model)
        print("torch.compile enabled (model compiled).")
    except Exception as e:
        print("torch.compile not enabled:", e)

# weight_decay set to 0.0 for early experiments (can re-enable later)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
total_steps = EPOCHS * STEPS_PER_EPOCH
# warmup increased to 5% of total_steps for stability without LayerNorm
warmup_steps = max(1, int(0.05 * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# --------------------------- Training loop with validation (grad accumulation) ---------------------------
best_val = float("inf")
epochs_no_improve = 0

print("Beginning training...")
for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    pbar = trange(STEPS_PER_EPOCH, desc=f"Epoch {ep}/{EPOCHS}")
    optimizer.zero_grad()
    start_epoch = time.time()
    for step in pbar:
        xb, yb = get_batch_from_concat(train_data)
        logits = model(xb, use_last_token=False)   # (B, T, V)
        B, T, V = logits.shape
        raw_loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))
        loss = raw_loss / GRAD_ACCUM_STEPS
        loss.backward()

        running += float(raw_loss.detach().cpu().item())

        # optimizer step on accumulation boundary
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(train_loss=running / (step + 1))

    epoch_time = time.time() - start_epoch
    train_avg = running / STEPS_PER_EPOCH

    # validation (reduced number of steps to save time)
    model.eval()
    vloss = 0.0
    val_steps = max(1, min(VAL_STEPS_LIMIT, len(val_data) // (BLOCK_SIZE * 10)))
    with torch.no_grad():
        for _ in range(val_steps):
            xb, yb = get_batch_from_concat(val_data)
            logits = model(xb, use_last_token=False)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))
            vloss += float(loss.item())
    val_avg = vloss / val_steps
    val_ppl = float(torch.exp(torch.tensor(val_avg))) if val_avg < 50 else float("inf")

    print(f"Epoch {ep}/{EPOCHS}  TrainLoss={train_avg:.4f}  ValLoss={val_avg:.4f}  ValPPL={val_ppl:.2f}  epoch_sec={epoch_time:.1f}")

    # checkpointing & early stopping (save try/except to avoid crash if disk issues)
    try:
        model.save(CKPT_PATH)
    except Exception as e:
        print("Warning: failed to save checkpoint:", e)

    if val_avg < best_val:
        best_val = val_avg
        try:
            model.save(BEST_CKPT_PATH)
            print("  New best val loss saved ->", BEST_CKPT_PATH)
        except Exception as e:
            print("Warning: failed to save best checkpoint:", e)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s) (patience={EARLY_STOPPING_PATIENCE})")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    # qualitative sample at reduced frequency
    if SAMPLE_EVERY > 0 and (ep % SAMPLE_EVERY == 0):
        model.eval()
        with torch.no_grad():
            prompt = "<|user|> Hello, how are you? <|assistant|> "
            enc = tokenizer.encode("System: You are SARAN. " + prompt, add_special_tokens=False)
            idx = torch.tensor([enc + [eos]], dtype=torch.long, device=DEVICE)
            if idx.shape[1] > model.block_size:
                idx = idx[:, -model.block_size:]
            out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            gen_ids = out[0].tolist()[idx.shape[1]:]
            if eos in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos)]
            sample = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            sample = " ".join(sample.split()).strip()
            print("Sample:", sample)
        model.train()

# ensure tokenizer present for inference
tokenizer.save_pretrained(TOKENIZER_DIR)
print("Training complete. Checkpoint ->", CKPT_PATH, "Tokenizer ->", TOKENIZER_DIR)