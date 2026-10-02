"""
model_utils.py — Model loading, tokenization, and metric computation
for the Banking77 6-class intent classifier.

Public API
----------
    build_tokenizer() -> DistilBertTokenizerFast
    tokenize_dataset(dataset, tokenizer) -> Dataset
    build_model(label2id, id2label) -> DistilBertForSequenceClassification
    compute_metrics(eval_pred) -> dict[str, float]
    get_data_collator(tokenizer) -> DataCollatorWithPadding

Run as a script (smoke test)
-----------------------------
    python -m src.model_utils

    Loads the tokenizer and model, runs a forward pass on a tiny batch,
    and reports logit shape, device, and GPU memory.
    Does NOT train anything.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from datasets import Dataset

from src.config import (
    MAX_LENGTH,
    MODEL_NAME,
    SELECTED_INTENTS,
)
from src.data_utils import build_label_mapping

NUM_LABELS = len(SELECTED_INTENTS)  # 6


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer() -> DistilBertTokenizerFast:
    """
    Load the DistilBERT fast tokenizer from the Hugging Face Hub.

    Uses the fast (Rust-backed) tokenizer for speed. No padding is applied
    here — padding is handled per-batch by DataCollatorWithPadding so that
    sequences are only padded to the longest example in each batch, not to
    MAX_LENGTH across the entire dataset.
    """
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset: Dataset, tokenizer: DistilBertTokenizerFast) -> Dataset:
    """
    Tokenize a dataset split.

    Design choices
    --------------
    - truncation=True   : clips at MAX_LENGTH (64) to protect VRAM.
    - padding=False     : no padding here; DataCollatorWithPadding pads
                          per-batch during training/evaluation.
    - return_tensors    : not set here; the collator converts to tensors.
    - The 'label' column is kept as-is (integer 0–5); the Trainer reads it.
    - 'text' and 'label_text' are removed after tokenization — they are not
      model inputs.

    Parameters
    ----------
    dataset   : A Dataset split with 'text' and 'label' columns.
    tokenizer : A loaded DistilBertTokenizerFast.

    Returns
    -------
    A Dataset with columns: input_ids, attention_mask, label.
    """
    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # pad per-batch, not per-dataset
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text", "label_text"],
        desc="Tokenizing",
    )
    return tokenized


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(
    label2id: dict[str, int] | None = None,
    id2label: dict[int, str] | None = None,
) -> DistilBertForSequenceClassification:
    """
    Load distilbert-base-uncased with a fresh classification head for 6 classes.

    The pretrained weights for the 6-layer DistilBERT encoder are loaded from
    the Hugging Face Hub. The classification head (a linear layer mapping the
    [CLS] token embedding → 6 logits) is randomly initialised — this is
    expected and triggers an 'ignore_mismatched_sizes'-style warning from
    transformers, which we suppress with ignore_mismatched_sizes=True.

    Parameters
    ----------
    label2id : dict mapping intent-string → integer label (0–5).
    id2label : dict mapping integer label → intent-string.
               Both default to the project's SELECTED_INTENTS mapping.

    Returns
    -------
    DistilBertForSequenceClassification on CPU (moved to GPU by Trainer).
    """
    if label2id is None or id2label is None:
        label2id, id2label = build_label_mapping(SELECTED_INTENTS)

    # Pass label mappings directly to from_pretrained so they are baked into
    # model.config — the Trainer and saved model will use these names.
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # head size changes from 2→6 labels
    )
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """
    Compute accuracy and macro-F1 from Trainer evaluation predictions.

    Called by Trainer after each evaluation step. Uses sklearn so the
    computation is identical to what evaluate.py will report on the test set.

    Parameters
    ----------
    eval_pred : transformers.EvalPrediction with .predictions and .label_ids.

    Returns
    -------
    {"accuracy": float, "f1": float}
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1":       float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

def get_data_collator(tokenizer: DistilBertTokenizerFast) -> DataCollatorWithPadding:
    """
    Return a DataCollatorWithPadding instance.

    This collator pads each batch to the longest sequence in that batch,
    rather than the global MAX_LENGTH. For Banking77 queries (avg ~10 tokens)
    this saves significant memory and speeds up training vs. static padding.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


# ---------------------------------------------------------------------------
# Script entry point: smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  MODEL SMOKE TEST")
    print("=" * 60)

    # --- Environment report ---
    import transformers
    print(f"\n  torch       : {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU         : {dev.name}")
        print(f"  VRAM total  : {total / 1024**3:.2f} GB")
        print(f"  VRAM free   : {free / 1024**3:.2f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load tokenizer ---
    print("\n  Loading tokenizer...")
    tokenizer = build_tokenizer()
    print(f"  Tokenizer class : {type(tokenizer).__name__}")
    print(f"  Vocab size      : {tokenizer.vocab_size:,}")

    # --- Verify tokenizer output on a sample sentence ---
    sample = "I am still waiting on my card."
    enc = tokenizer(sample, truncation=True, max_length=MAX_LENGTH, padding=False)
    print(f"\n  Sample: {sample!r}")
    print(f"  Token keys      : {list(enc.keys())}")
    print(f"  input_ids       : {enc['input_ids']}")
    print(f"  attention_mask  : {enc['attention_mask']}")
    assert "input_ids" in enc and "attention_mask" in enc, \
        "Tokenizer missing expected keys"
    assert "token_type_ids" not in enc, \
        "DistilBERT should not produce token_type_ids"

    # --- Load model ---
    print("\n  Loading model...")
    label2id, id2label = build_label_mapping(SELECTED_INTENTS)
    model = build_model(label2id, id2label)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model class     : {type(model).__name__}")
    print(f"  Parameters      : {n_params:,}")
    print(f"  Device          : {next(model.parameters()).device}")
    print(f"  Config num_labels: {model.config.num_labels}")
    print(f"  Config id2label : {model.config.id2label}")

    # Verify config consistency
    assert model.config.num_labels == NUM_LABELS, \
        f"Expected num_labels={NUM_LABELS}, got {model.config.num_labels}"
    assert model.config.id2label == id2label, \
        "Model config id2label does not match build_label_mapping output"
    assert model.config.label2id == label2id, \
        "Model config label2id does not match build_label_mapping output"

    # --- Forward pass smoke test ---
    print("\n  Running forward pass on a batch of 4 synthetic sentences...")
    BATCH = [
        "I am still waiting on my card.",
        "What is today's exchange rate?",
        "I want a refund please.",
        "My card was stolen yesterday.",
    ]
    BATCH_SIZE = len(BATCH)

    inputs = tokenizer(
        BATCH,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,          # pad to longest in this tiny batch
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print(f"  Logit shape     : {tuple(logits.shape)}")
    print(f"  Expected shape  : ({BATCH_SIZE}, {NUM_LABELS})")
    print(f"  All finite      : {torch.isfinite(logits).all().item()}")
    print(f"  Logits (raw)    :\n{logits.cpu()}")

    assert logits.shape == (BATCH_SIZE, NUM_LABELS), \
        f"Shape mismatch: got {tuple(logits.shape)}, expected ({BATCH_SIZE}, {NUM_LABELS})"
    assert torch.isfinite(logits).all(), \
        "Non-finite values in logits (NaN or Inf)"

    # Check predicted labels are valid
    predicted_ids = logits.argmax(dim=-1).cpu().tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]
    print(f"\n  Predicted labels (untrained, random): {predicted_labels}")
    assert all(0 <= i < NUM_LABELS for i in predicted_ids), \
        "Predicted label ID out of range"

    # GPU memory after model load
    if torch.cuda.is_available():
        free_after, total_after = torch.cuda.mem_get_info(0)
        used = (total_after - free_after) / 1024**3
        print(f"\n  VRAM used after model load: {used:.3f} GB")

    print("\n  Smoke test PASSED ✓")
    print("=" * 60)
