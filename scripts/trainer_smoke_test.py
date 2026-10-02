"""
scripts/trainer_smoke_test.py
------------------------------
Short training smoke test using the real HuggingFace Trainer.

What it does
------------
  - Loads the real mteb/banking77 6-class subset via prepare_data()
  - Takes a tiny balanced subset: 4 examples/class train (24), 2 examples/class val (12)
  - Tokenizes with the real distilbert tokenizer
  - Runs exactly 3 gradient steps with FP16 on the GPU
  - Triggers one evaluation pass (after step 3)
  - Saves one checkpoint
  - Reports: step losses, eval loss, eval accuracy, eval F1, GPU memory, checkpoint files

What it does NOT do
-------------------
  - Does not start full training
  - Does not touch the held-out test split
  - Does not modify src/ or any existing file

Usage
-----
  python -m scripts.trainer_smoke_test
  python scripts/trainer_smoke_test.py

If FP16 fails
-------------
  The script catches the error, reports it clearly, and re-runs with fp16=False.
  No silent fallbacks.
"""

from __future__ import annotations

import os
import random
import shutil
from collections import Counter, defaultdict

import torch
from transformers import Trainer, TrainingArguments

from src.config import SEED
from src.data_utils import build_label_mapping, prepare_data
from src.model_utils import (
    build_model,
    build_tokenizer,
    compute_metrics,
    get_data_collator,
    tokenize_dataset,
)

# ---------------------------------------------------------------------------
# Smoke-test constants (intentionally tiny)
# ---------------------------------------------------------------------------
SMOKE_OUTPUT_DIR = "outputs/trainer-smoke-test"
TRAIN_PER_CLASS  = 4    # 4 × 6 = 24 train examples
VAL_PER_CLASS    = 2    # 2 × 6 = 12 val examples
MAX_STEPS        = 3    # gradient steps to run
BATCH_SIZE       = 4    # per-device batch size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_per_class(dataset, n_per_class: int, seed: int = SEED):
    """Return a balanced subset: exactly n_per_class examples per label."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(dataset["label"]):
        by_class[lbl].append(i)

    rng = random.Random(seed)
    selected: list[int] = []
    for lbl in sorted(by_class):
        pool = by_class[lbl][:]
        rng.shuffle(pool)
        selected.extend(pool[:n_per_class])

    return dataset.select(sorted(selected))


def _build_trainer(fp16: bool, tokenizer, train_tok, val_tok, model) -> Trainer:
    """Construct a Trainer with the smoke-test configuration."""
    if os.path.exists(SMOKE_OUTPUT_DIR):
        shutil.rmtree(SMOKE_OUTPUT_DIR)

    args = TrainingArguments(
        output_dir=SMOKE_OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=MAX_STEPS,          # one eval after the final step
        save_strategy="steps",
        save_steps=MAX_STEPS,          # one checkpoint after the final step
        load_best_model_at_end=False,  # smoke test only
        learning_rate=2e-5,
        warmup_steps=0,
        weight_decay=0.01,
        fp16=fp16,
        logging_steps=1,               # log loss at every step
        dataloader_num_workers=0,      # avoid multiprocessing with tiny data
        report_to="none",
        seed=SEED,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=get_data_collator(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,    # 'processing_class' is the transformers 5.x name
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 64
    print(sep)
    print("  TRAINER SMOKE TEST")
    print(sep)

    # ── Environment ─────────────────────────────────────────────────────────
    import transformers
    cuda_ok = torch.cuda.is_available()
    print(f"\n  torch        : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    if cuda_ok:
        dev = torch.cuda.get_device_properties(0)
        free0, total = torch.cuda.mem_get_info(0)
        print(f"  GPU          : {dev.name}")
        print(f"  VRAM total   : {total / 1024**3:.2f} GB")
        print(f"  VRAM free    : {free0 / 1024**3:.2f} GB")

    # ── 1. Data ─────────────────────────────────────────────────────────────
    print(f"\n  [1/5] Preparing data...")
    data     = prepare_data()
    label2id = data["label2id"]
    id2label = data["id2label"]

    train_sub = _select_per_class(data["train"], TRAIN_PER_CLASS)
    val_sub   = _select_per_class(data["val"],   VAL_PER_CLASS)

    for name, sub in [("train", train_sub), ("val", val_sub)]:
        counts = Counter(sub["label"])
        assert len(set(counts.values())) == 1, f"{name} subset not balanced: {dict(counts)}"

    print(f"  Smoke train  : {len(train_sub)} examples  ({TRAIN_PER_CLASS}/class)")
    print(f"  Smoke val    : {len(val_sub)} examples  ({VAL_PER_CLASS}/class)")

    # ── 2. Tokenize ─────────────────────────────────────────────────────────
    print(f"\n  [2/5] Tokenizing...")
    tokenizer = build_tokenizer()
    train_tok = tokenize_dataset(train_sub, tokenizer)
    val_tok   = tokenize_dataset(val_sub,   tokenizer)

    # DistilBertForSequenceClassification.forward() expects 'labels' (plural).
    # Our data pipeline stores 'label' (singular) — rename before passing to Trainer.
    # NOTE: train.py will need the same rename.
    train_tok = train_tok.rename_column("label", "labels")
    val_tok   = val_tok.rename_column("label",   "labels")
    print(f"  Columns      : {train_tok.column_names}")

    # ── 3. Model ────────────────────────────────────────────────────────────
    print(f"\n  [3/5] Loading model...")
    if cuda_ok:
        free_pre, _ = torch.cuda.mem_get_info(0)

    model = build_model(label2id, id2label)

    if cuda_ok:
        free_post, _ = torch.cuda.mem_get_info(0)
        print(f"  VRAM for model : {(free_pre - free_post)/1024**3:.3f} GB")
        print(f"  VRAM remaining : {free_post/1024**3:.2f} GB")

    # ── 4. Train ────────────────────────────────────────────────────────────
    print(f"\n  [4/5] Training  ({MAX_STEPS} steps, batch={BATCH_SIZE}, fp16={cuda_ok})...")

    fp16 = cuda_ok
    try:
        trainer = _build_trainer(fp16, tokenizer, train_tok, val_tok, model)
        train_result = trainer.train()
    except RuntimeError as exc:
        print(f"\n  ⚠  FP16 training FAILED: {exc}")
        print(f"  Retrying with fp16=False ...")
        fp16 = False
        model = build_model(label2id, id2label)   # fresh model; grads may be corrupt
        trainer = _build_trainer(fp16, tokenizer, train_tok, val_tok, model)
        train_result = trainer.train()
        print(f"  FP32 retry succeeded.")

    # ── 5. Results ──────────────────────────────────────────────────────────
    print(f"\n  [5/5] Collecting results...")

    m = train_result.metrics
    train_loss = float(m.get("train_loss", float("nan")))

    print(f"\n{sep}")
    print(f"  RESULTS")
    print(sep)

    print(f"\n  ── Training ──────────────────────────────────────────────")
    print(f"  Steps completed  : {train_result.global_step}")
    print(f"  FP16 used        : {fp16}")
    print(f"  Train loss       : {train_loss:.6f}")
    print(f"  Runtime (s)      : {m.get('train_runtime', 0):.2f}")
    print(f"  Samples/sec      : {m.get('train_samples_per_second', 0):.2f}")

    assert torch.isfinite(torch.tensor(train_loss)).item(), \
        f"Train loss is not finite: {train_loss}"
    print(f"  Loss finite      : ✓")

    # Explicit validation pass (separate from the Trainer's internal step-eval)
    print(f"\n  ── Evaluation  (val set, {len(val_tok)} examples) ──────────")
    eval_metrics = trainer.evaluate()
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k:<34}: {v:.6f}" if isinstance(v, float) else f"  {k:<34}: {v}")

    eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
    assert torch.isfinite(torch.tensor(eval_loss)).item(), \
        f"Eval loss is not finite: {eval_loss}"
    print(f"  Eval loss finite : ✓")

    # Checkpoint verification
    print(f"\n  ── Checkpoint ────────────────────────────────────────────")
    ckpts = sorted(d for d in os.listdir(SMOKE_OUTPUT_DIR) if d.startswith("checkpoint-"))
    assert ckpts, f"No checkpoint-* dir found in {SMOKE_OUTPUT_DIR}"
    ckpt_path  = os.path.join(SMOKE_OUTPUT_DIR, ckpts[-1])
    ckpt_files = sorted(os.listdir(ckpt_path))
    print(f"  Dir              : {ckpt_path}")
    print(f"  Files            : {ckpt_files}")
    has_weights = "model.safetensors" in ckpt_files or "pytorch_model.bin" in ckpt_files
    assert has_weights, "No model weights found in checkpoint"
    print(f"  Weights present  : ✓")
    print(f"  config.json      : {'✓' if 'config.json' in ckpt_files else '⚠ missing'}")

    # GPU memory after training
    if cuda_ok:
        free_end, total_vram = torch.cuda.mem_get_info(0)
        print(f"\n  ── GPU Memory After Training ─────────────────────────")
        print(f"  VRAM total       : {total_vram/1024**3:.2f} GB")
        print(f"  VRAM used        : {(total_vram - free_end)/1024**3:.3f} GB")
        print(f"  VRAM free        : {free_end/1024**3:.2f} GB")

    print(f"\n  Smoke test PASSED ✓")
    print(sep)


if __name__ == "__main__":
    main()
