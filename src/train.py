"""
src/train.py — Full training pipeline for the Banking77 6-class intent classifier.

Orchestrates: data preparation → tokenization → model loading → Trainer setup
→ (optional) mini smoke test → full training → best-model save.

Public API
----------
    build_training_args(num_train_examples) -> TrainingArguments
    train(run_smoke_test=True) -> TrainOutput

Run as a script
---------------
    python -m src.train

    Runs the full 3-epoch fine-tuning on the prepared train/val splits.
    The best checkpoint (by eval_loss) is saved to OUTPUT_DIR.
    The held-out test split is never used here.
"""

from __future__ import annotations

import math
import os
import random
import shutil
from collections import Counter, defaultdict

import torch
from transformers import Trainer, TrainingArguments

from src.config import (
    BATCH_SIZE,
    FP16,
    LEARNING_RATE,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    SELECTED_INTENTS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.data_utils import prepare_data
from src.model_utils import (
    build_model,
    build_tokenizer,
    compute_metrics,
    get_data_collator,
    tokenize_dataset,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_per_class(dataset, n_per_class: int, seed: int = SEED):
    """Return a balanced subset of n_per_class examples per label."""
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


def _rename_label_column(dataset):
    """
    Rename 'label' → 'labels'.

    DistilBertForSequenceClassification.forward() expects the column to be
    named 'labels' (plural). Our data pipeline stores it as 'label' (singular).
    This rename is the only place where that translation happens; it must be
    applied to every split before it is passed to the Trainer.
    """
    return dataset.rename_column("label", "labels")


# ---------------------------------------------------------------------------
# TrainingArguments factory
# ---------------------------------------------------------------------------

def build_training_args(
    num_train_examples: int,
    output_dir: str = OUTPUT_DIR,
    *,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
    weight_decay: float = WEIGHT_DECAY,
    fp16: bool = FP16,
    seed: int = SEED,
) -> TrainingArguments:
    """
    Build a TrainingArguments object from the project configuration.

    API notes (transformers 5.18.0)
    --------------------------------
    - 'warmup_ratio' does not exist in this version; only 'warmup_steps' is
      supported. We compute warmup_steps = ceil(warmup_ratio * total_steps)
      here so the config value is honoured exactly.
    - 'eval_strategy' is the correct name ('evaluation_strategy' is the old
      deprecated alias; do not use it).
    - 'save_strategy' must equal 'eval_strategy' when load_best_model_at_end=True.
    - 'metric_for_best_model' must match a key returned by compute_metrics().
      We use "eval_loss" (always present) rather than "eval_f1" so that Trainer
      uses the built-in loss tracking without requiring a custom metric key.

    Parameters
    ----------
    num_train_examples : Total number of training examples (used for warmup).
    output_dir         : Directory for checkpoints and the final best model.
    """
    steps_per_epoch = math.ceil(num_train_examples / batch_size)
    total_steps     = steps_per_epoch * num_epochs
    warmup_steps    = math.ceil(warmup_ratio * total_steps)

    return TrainingArguments(
        output_dir=output_dir,

        # Training schedule
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        # Evaluation and checkpointing
        # Both strategies must be identical for load_best_model_at_end=True.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,           # lower eval_loss is better

        # Mixed precision
        fp16=fp16 and torch.cuda.is_available(),

        # Logging
        logging_steps=10,
        report_to="none",

        seed=seed,
    )


# ---------------------------------------------------------------------------
# Smoke test (tiny subset, few steps)
# ---------------------------------------------------------------------------

_SMOKE_OUTPUT_DIR   = "outputs/train-smoke-test"
_SMOKE_PER_CLASS    = 4    # 4 × 6 = 24 train examples
_SMOKE_VAL_PER_CLASS = 2   # 2 × 6 = 12 val examples
_SMOKE_MAX_STEPS    = 2    # two gradient steps are enough to confirm pipeline


def run_smoke_test(
    train_data,
    val_data,
    label2id: dict,
    id2label: dict,
    tokenizer,
) -> None:
    """
    Run a short Trainer smoke test on a tiny balanced subset.

    Verifies that one forward pass, one backward pass, and one evaluation
    pass complete without error. Raises AssertionError or RuntimeError if
    anything goes wrong. Does NOT save a persistent best-model.

    Parameters
    ----------
    train_data, val_data : Full (not-yet-tokenized) train/val Dataset splits.
    label2id, id2label   : Label mappings from prepare_data().
    tokenizer            : Loaded DistilBertTokenizerFast.
    """
    print("\n  ── Smoke test ────────────────────────────────────────────")
    print(f"  Subset: {_SMOKE_PER_CLASS}/class train, {_SMOKE_VAL_PER_CLASS}/class val")

    # Tiny balanced subsets
    sub_train = _select_per_class(train_data, _SMOKE_PER_CLASS)
    sub_val   = _select_per_class(val_data,   _SMOKE_VAL_PER_CLASS)

    # Tokenize and rename
    tok_train = _rename_label_column(tokenize_dataset(sub_train, tokenizer))
    tok_val   = _rename_label_column(tokenize_dataset(sub_val,   tokenizer))

    # Fresh model for the smoke test (isolated from the full training run)
    smoke_model = build_model(label2id, id2label)

    if os.path.exists(_SMOKE_OUTPUT_DIR):
        shutil.rmtree(_SMOKE_OUTPUT_DIR)

    smoke_args = TrainingArguments(
        output_dir=_SMOKE_OUTPUT_DIR,
        max_steps=_SMOKE_MAX_STEPS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=_SMOKE_MAX_STEPS,
        save_strategy="steps",
        save_steps=_SMOKE_MAX_STEPS,
        load_best_model_at_end=False,
        learning_rate=LEARNING_RATE,
        warmup_steps=0,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16 and torch.cuda.is_available(),
        logging_steps=1,
        report_to="none",
        seed=SEED,
    )

    smoke_trainer = Trainer(
        model=smoke_model,
        args=smoke_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=get_data_collator(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    result = smoke_trainer.train()
    loss   = float(result.metrics.get("train_loss", float("nan")))
    assert math.isfinite(loss), f"Smoke test: train loss is not finite ({loss})"

    eval_out = smoke_trainer.evaluate()
    eval_loss = float(eval_out.get("eval_loss", float("nan")))
    assert math.isfinite(eval_loss), f"Smoke test: eval loss is not finite ({eval_loss})"

    print(f"  train_loss : {loss:.4f}  ✓")
    print(f"  eval_loss  : {eval_loss:.4f}  ✓")
    print(f"  eval_acc   : {eval_out.get('eval_accuracy', 'N/A')}")
    print(f"  eval_f1    : {eval_out.get('eval_f1', 'N/A')}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        used = (total - free) / 1024 ** 3
        print(f"  VRAM used  : {used:.3f} GB  (free: {free/1024**3:.2f} GB)")

    print("  Smoke test PASSED ✓")


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def train(run_smoke_test_first: bool = True) -> object:
    """
    Run the full training pipeline.

    Steps
    -----
    1. prepare_data()   — load, filter, split, leakage check
    2. Tokenize train and val splits; rename 'label' → 'labels'
    3. (optional) run a 2-step smoke test to validate the pipeline
    4. Build TrainingArguments from config constants
    5. Trainer.train() for NUM_EPOCHS epochs
    6. Save the best model and tokenizer to OUTPUT_DIR

    The held-out test split is not used at any point here.

    Returns
    -------
    transformers.trainer_utils.TrainOutput  (metrics, global_step, etc.)
    """
    sep = "=" * 64
    print(sep)
    print("  TRAINING PIPELINE")
    print(sep)

    # -- Environment report --------------------------------------------------
    import transformers
    cuda = torch.cuda.is_available()
    print(f"\n  torch        : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    print(f"  fp16 config  : {FP16}  (active: {FP16 and cuda})")
    if cuda:
        dev = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU          : {dev.name}")
        print(f"  VRAM total   : {total/1024**3:.2f} GB")
        print(f"  VRAM free    : {free/1024**3:.2f} GB")

    # -- 1. Data -------------------------------------------------------------
    print(f"\n  [1/5] Preparing data...")
    data     = prepare_data()
    label2id = data["label2id"]
    id2label = data["id2label"]
    # test split is intentionally not used here
    train_raw = data["train"]
    val_raw   = data["val"]

    print(f"  Train : {len(train_raw)} examples")
    print(f"  Val   : {len(val_raw)} examples")

    # -- 2. Tokenizer --------------------------------------------------------
    print(f"\n  [2/5] Loading tokenizer...")
    tokenizer = build_tokenizer()

    # -- 3. Smoke test -------------------------------------------------------
    if run_smoke_test_first:
        print(f"\n  [3/5] Running pre-training smoke test...")
        run_smoke_test(train_raw, val_raw, label2id, id2label, tokenizer)
    else:
        print(f"\n  [3/5] Smoke test skipped (run_smoke_test_first=False).")

    # -- 4. Tokenize full splits ---------------------------------------------
    print(f"\n  [4/5] Tokenizing full splits...")
    train_tok = _rename_label_column(tokenize_dataset(train_raw, tokenizer))
    val_tok   = _rename_label_column(tokenize_dataset(val_raw,   tokenizer))
    print(f"  Columns : {train_tok.column_names}")

    # -- 5. Build Trainer and train ------------------------------------------
    print(f"\n  [5/5] Training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE})...")

    if cuda:
        free_pre, _ = torch.cuda.mem_get_info(0)

    model = build_model(label2id, id2label)

    if cuda:
        free_post, _ = torch.cuda.mem_get_info(0)
        print(f"  VRAM for model      : {(free_pre - free_post)/1024**3:.3f} GB")
        print(f"  VRAM remaining      : {free_post/1024**3:.2f} GB")

    args = build_training_args(
        num_train_examples=len(train_tok),
        output_dir=OUTPUT_DIR,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=get_data_collator(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    train_output = trainer.train()

    # -- Report training results ---------------------------------------------
    m = train_output.metrics
    print(f"\n{sep}")
    print(f"  TRAINING COMPLETE")
    print(sep)
    print(f"  Steps          : {train_output.global_step}")
    print(f"  Epochs         : {NUM_EPOCHS}")
    print(f"  Train loss     : {m.get('train_loss', 'N/A'):.6f}")
    print(f"  Runtime (s)    : {m.get('train_runtime', 0):.1f}")
    print(f"  Samples/sec    : {m.get('train_samples_per_second', 0):.1f}")

    # -- Save best model + tokenizer -----------------------------------------
    print(f"\n  Saving best model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Saved ✓")

    # Final GPU memory report
    if cuda:
        free_end, total_vram = torch.cuda.mem_get_info(0)
        print(f"\n  VRAM used : {(total_vram - free_end)/1024**3:.3f} GB")
        print(f"  VRAM free : {free_end/1024**3:.2f} GB")

    return train_output


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(run_smoke_test_first=True)
