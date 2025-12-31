#!/usr/bin/env python3

"""
ChatSARAN-Train.py — full, self-contained training script (strict single-block SARAN)

Features:
 - Strict SARAN single-block implementation with explicit step numbering (Steps 1..15).
 - Small scaled residual injection at step 11.5: x = x + residual_scale * attn_out.
 - Chunked tokenization that splits long encodings into BLOCK_SIZE chunks to avoid
   tokenizer/model length warnings.
 - Data building from streaming HF datasets into data_ids.pt (rebuilds if missing or
   smaller than MAX_TOKENS).
 - Environment-variable overrides for fast experiments: LR, D_MODEL, BATCH_SIZE,
   GRAD_ACCUM_STEPS, RESIDUAL_SCALE, STEPS_PER_EPOCH, EPOCHS, SMOKE, USE_TORCH_COMPILE, USE_AMP.
 - Safety: NaN/Inf guards, grad clipping, optional AMP on CUDA, optional torch.compile.
 - Checkpointing (regular + best), tokenizer saving.
Usage:
 - Edit defaults at top or set env vars, e.g.:
     LR=5e-5 RESIDUAL_SCALE=0.05 SMOKE=1 SMOKE_STEPS=200 SMOKE_EPOCHS=3 python ChatSARAN-Train.py
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

# -------------------------- env helpers ---------------------------------------
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

# --------------------------- HYPERPARAMS (defaults, overridable) --------------
MAX_TOKENS = env_int("MAX_TOKENS", 10_000_000)
BLOCK_SIZE = env_int("BLOCK_SIZE", 256)

D_MODEL = env_int("D_MODEL", 384)            # increase if you have extra memory (512)
BATCH_SIZE = env_int("BATCH_SIZE", 32)
GRAD_ACCUM_STEPS = env_int("GRAD_ACCUM_STEPS", 4)

# By default compute STEPS_PER_EPOCH from MAX_TOKENS; override with env if needed
DEFAULT_STEPS = max(1, int(MAX_TOKENS // (BATCH_SIZE * BLOCK_SIZE)))
STEPS_PER_EPOCH = env_int("STEPS_PER_EPOCH", DEFAULT_STEPS)

# Smoke-run overrides (for quick experiments)
SMOKE = env_bool("SMOKE", False)
if SMOKE:
    STEPS_PER_EPOCH = env_int("SMOKE_STEPS", min(200, STEPS_PER_EPOCH))
    EPOCHS = env_int("EPOCHS", env_int("SMOKE_EPOCHS", 3))
else:
    EPOCHS = env_int("EPOCHS", 10000)

LR = env_float("LR", 5e-5)                   # default safe LR for no-LN
RESIDUAL_SCALE = env_float("RESIDUAL_SCALE", 0.1)
VAL_SPLIT = env_float("VAL_SPLIT", 0.05)

EARLY_STOPPING_PATIENCE = env_int("EARLY_STOPPING_PATIENCE", 20)
VAL_STEPS_LIMIT = env_int("VAL_STEPS_LIMIT", 50)
SAMPLE_EVERY = env_int("SAMPLE_EVERY", 5)   # 0 disables sample printing

# optimizer/scheduler
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.0)   # 0.0 for early experiments
BETA1 = env_float("BETA1", 0.9)
BETA2 = env_float("BETA2", 0.95)

# compile + precision
USE_TORCH_COMPILE = env_bool("USE_TORCH_COMPILE", False)
USE_AMP = env_bool("USE_AMP", True)  # only used when CUDA available

# generation debug settings
TEMPERATURE = env_float("TEMPERATURE", 0.20)
TOP_K = env_int("TOP_K", 40)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 128)
# -------------------------------------------------------------------------------

DATA_IDS = "data_ids.pt"
CKPT_PATH = "chat_saran_ckpt.pt"
BEST_CKPT_PATH = "chat_saran_ckpt.best.pt"
TOKENIZER_DIR = "chat_saran_tokenizer"

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "mps":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# --------------------------- Prepare tokenizer + data (chunking) ---------------
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
    tokenizer.model_max_length = 10**6  # avoid noisy warnings while encoding then chunking
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    ids = []
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

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
                inst = (ex.get("instruction") or ex.get("prompt") or "")[:2000]
                resp = (ex.get("response") or ex.get("completion") or "")[:2000]
                if inst or resp:
                    piece = "<|user|> " + inst + " <|assistant|> " + resp
                else:
                    text = (ex.get("text") or ex.get("article") or "")
                    if isinstance(text, list):
                        text = " ".join(text)
                    piece = (text or "")[:2000]

                if not piece:
                    continue

                tokens = tokenizer.encode(piece, add_special_tokens=False)
                if len(tokens) <= BLOCK_SIZE:
                    ids.extend(tokens + [eos])
                else:
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
            print(f"  skipping dataset {ds_name}: {e}")
        if len(ids) >= MAX_TOKENS:
            break

    ids = ids[:MAX_TOKENS]
    torch.save(ids, DATA_IDS)
    print("Saved processed token ids ->", DATA_IDS, " (tokens):", len(ids))
else:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokenizer.model_max_length = 10**6
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("", add_special_tokens=False)[0]

data = torch.tensor(ids, dtype=torch.long)  # keep on CPU
vocab_size = tokenizer.vocab_size
print(f"Built token stream: {len(data)} tokens | vocab_size={vocab_size}")

n_train = int(len(data) * (1.0 - VAL_SPLIT))
train_data = data[:n_train]
val_data = data[n_train:]
print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

# --------------------------- Model (strict single-block SARAN) ------------------
class ChatSARAN(nn.Module):
    """
    Strict single-block SARAN with explicit step numbering (1..15).
    Only modification: scaled residual injection at step 11.5.
    """
    def __init__(self, vocab_size: int, d_model: int = D_MODEL, block_size: int = BLOCK_SIZE, residual_scale: float = RESIDUAL_SCALE):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.residual_scale = float(residual_scale)

        # Step 1: token embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Step 2: positional embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)
        # Steps 5-7: Q/K/V projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)  # Step 5
        self.Wk = nn.Linear(d_model, d_model, bias=False)  # Step 6
        self.Wv = nn.Linear(d_model, d_model, bias=False)  # Step 7
        # Step 8: scaling constant
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(d_model), dtype=torch.float32))
        # Step 9: causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
        # Steps 13-14: output projection
        self.Wout = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, idx: torch.LongTensor, use_last_token: bool = False):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        # Steps 1-4
        # Step 1: token embeddings
        tok = self.tok_emb(idx)                                      # (B, T, d)
        # Step 2: positional embeddings
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))  # (1, T, d)
        # Step 3-4: combine
        x = tok + pos                                                 # (B, T, d)

        # Steps 5-7: compute q, k, v
        q = self.Wq(x)                                                # (B, T, d)
        k = self.Wk(x)                                                # (B, T, d)
        v = self.Wv(x)                                                # (B, T, d)

        # Step 8: dot-product scaling
        scores = (q @ k.transpose(-2, -1)) * self.scale               # (B, T, T)
        # Step 9: causal mask
        m = self.mask[:T, :T].to(scores.device)
        scores = scores.masked_fill(~m, float("-inf"))
        # Step 10: softmax -> attention weights
        attn = F.softmax(scores, dim=-1)                              # (B, T, T)
        # Step 11: attention output
        attn_out = attn @ v                                           # (B, T, d)

        # Step 11.5: scaled residual injection (preserves SARAN flow)
        x = x + self.residual_scale * attn_out                        # (B, T, d)

        # Step 12: last token rep
        last = x[:, -1, :]                                            # (B, d)

        # Steps 13-15: output projection
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
        rs = ckpt.get("residual_scale", RESIDUAL_SCALE)
        model = cls(ckpt["vocab_size"], ckpt["d_model"], ckpt["block_size"], residual_scale=rs)
        model.load_state_dict(ckpt["state_dict"])
        return model

# --------------------------- Helpers ------------------------------------------------
def get_batch_from_concat(data_tensor: torch.LongTensor, block_size: int = BLOCK_SIZE,
                          batch_size: int = BATCH_SIZE, device=DEVICE) -> Tuple[torch.LongTensor, torch.LongTensor]:
    n = data_tensor.size(0)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters())

# --------------------------- Instantiate + optimizer + scheduler ---------------------
model = ChatSARAN(vocab_size=vocab_size, d_model=D_MODEL, block_size=BLOCK_SIZE, residual_scale=RESIDUAL_SCALE).to(DEVICE)

# optional compile (try/catch)
if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model)
        print("torch.compile enabled.")
    except Exception as e:
        print("torch.compile not enabled:", e)

# optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
total_steps = max(1, EPOCHS * STEPS_PER_EPOCH)
warmup_steps = max(1, int(0.05 * total_steps))  # 5% warmup default
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# mixed precision (CUDA) setup
use_amp = (DEVICE.type == "cuda") and USE_AMP
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

print(f"Model params: {count_params(model):,}  D_MODEL={D_MODEL}  residual_scale={RESIDUAL_SCALE}")
print(f"Steps/epoch: {STEPS_PER_EPOCH}  Batch={BATCH_SIZE}  GradAcc={GRAD_ACCUM_STEPS}  LR={LR}")
print(f"Total steps={total_steps}  warmup_steps={warmup_steps}  use_amp={use_amp}")

# --------------------------- Training loop with safety checks --------------------
best_val = float("inf")
epochs_no_improve = 0

def is_finite_tensor(t: torch.Tensor):
    return torch.isfinite(t).all().item()

print("Beginning training...")
for ep in range(1, EPOCHS + 1):
    # print current LR
    for pg in optimizer.param_groups:
        print(f"Epoch {ep} starting LR: {pg['lr']:.6g}")

    model.train()
    running = 0.0
    pbar = trange(STEPS_PER_EPOCH, desc=f"Epoch {ep}/{EPOCHS}")
    optimizer.zero_grad()
    start_epoch = time.time()

    for step in pbar:
        xb, yb = get_batch_from_concat(train_data)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb, use_last_token=False)
            B, T, V = logits.shape
            raw_loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))

        # NaN/Inf guard (abort early with diagnostics)
        if not torch.isfinite(raw_loss):
            print("ERROR: raw_loss is not finite (NaN/Inf). Dumping diagnostics and aborting.")
            print("loss:", raw_loss)
            try:
                with torch.no_grad():
                    mx = logits.max().item() if is_finite_tensor(logits) else float("inf")
                    mn = logits.min().item() if is_finite_tensor(logits) else float("-inf")
                print(f"logits min/max: {mn}/{mx}")
            except Exception as e:
                print("Failed to inspect logits:", e)
            try:
                model.save("chat_saran_ckpt.nan_abort.pt")
                print("Saved model to chat_saran_ckpt.nan_abort.pt")
            except Exception as e:
                print("Failed to save model:", e)
            raise RuntimeError("NaN detected in loss; aborting training.")

        loss = raw_loss / GRAD_ACCUM_STEPS
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += float(raw_loss.detach().cpu().item())

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(train_loss=running / (step + 1))

    epoch_time = time.time() - start_epoch
    train_avg = running / STEPS_PER_EPOCH

    # lightweight validation
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

    # per-epoch gradient norm summary
    total_grad_norm = 0.0
    n_g = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += float(p.grad.detach().norm().cpu().item())
            n_g += 1
    avg_grad_norm = total_grad_norm / max(1, n_g)

    print(f"Epoch {ep}/{EPOCHS}  TrainLoss={train_avg:.4f}  ValLoss={val_avg:.4f}  ValPPL={val_ppl:.2f}  epoch_sec={epoch_time:.1f}  avg_grad_norm={avg_grad_norm:.6g}")

    # checkpoint & early stopping
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

    # qualitative sample (reduced frequency)
    if SAMPLE_EVERY > 0 and (ep % SAMPLE_EVERY == 0):
        model.eval()
        with torch.no_grad():
            prompt = "<|user|> Hello, how are you? <|assistant|> "
            enc = tokenizer.encode("System: You are SARAN. " + prompt, add_special_tokens=False)
            idx = torch.tensor([enc + [eos]], dtype=torch.long, device=DEVICE)
            if idx.shape[1] > model.block_size:
                idx = idx[:, -model.block_size:]
            try:
                out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
                gen_ids = out[0].tolist()[idx.shape[1]:]
                if eos in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos)]
                sample = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sample = " ".join(sample.split()).strip()
                print("Sample:", sample)
            except Exception as e:
                print("Sample generation failed:", e)
        model.train()

# save tokenizer & final message
tokenizer.save_pretrained(TOKENIZER_DIR)
print("Training complete. Checkpoint ->", CKPT_PATH, " Best ->", BEST_CKPT_PATH)