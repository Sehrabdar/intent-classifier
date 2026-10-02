"""
src/evaluate.py — Final evaluation of the best saved model on the held-out test set.

This module is intentionally separate from training. It:
  - Loads the saved model and tokenizer from OUTPUT_DIR (never retrains).
  - Obtains the test split from prepare_data() (the same 240 examples held
    out since Milestone 1; they were never passed to the Trainer).
  - Runs a single, one-shot evaluation pass. The test set must not be used
    for hyperparameter tuning or evaluated more than once.
  - Reports overall accuracy, macro-F1, per-class precision/recall/F1,
    and a confusion matrix with intent names on both axes.

Public API
----------
    load_saved_model(model_dir) -> tuple[model, tokenizer]
    evaluate_on_test(model, tokenizer, test_dataset, id2label) -> EvalResult
    format_report(result) -> str
    run_evaluation(model_dir) -> EvalResult

Run as a script
---------------
    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.config import BATCH_SIZE, OUTPUT_DIR, SELECTED_INTENTS
from src.data_utils import build_label_mapping, prepare_data
from src.model_utils import get_data_collator, tokenize_dataset


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    All evaluation outputs in one place.

    Attributes
    ----------
    y_true       : Ground-truth integer labels (length = test set size).
    y_pred       : Predicted integer labels.
    label_names  : Intent name strings in label-ID order (index = label ID).
    accuracy     : Overall accuracy (0–1).
    macro_f1     : Macro-averaged F1 across all classes.
    report_dict  : sklearn classification_report as a nested dict.
    conf_matrix  : numpy confusion matrix, shape (num_classes, num_classes).
    """
    y_true:      np.ndarray
    y_pred:      np.ndarray
    label_names: list[str]
    accuracy:    float
    macro_f1:    float
    report_dict: dict
    conf_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_saved_model(
    model_dir: str = OUTPUT_DIR,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """
    Load the best saved model and its tokenizer from model_dir.

    This reads the weights saved by trainer.save_model() and does NOT touch
    any checkpoint subfolder or re-download the pretrained base model.

    Parameters
    ----------
    model_dir : Path to the directory produced by trainer.save_model().
                Defaults to OUTPUT_DIR from config.py.

    Returns
    -------
    (model, tokenizer)  — model is on CPU; inference code moves to GPU if needed.
    """
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    test_dataset,
    id2label: dict[int, str],
) -> EvalResult:
    """
    Run one evaluation pass on test_dataset and return an EvalResult.

    Uses a read-only Trainer (no training arguments that affect training) to
    run batched inference via Trainer.predict(), which handles GPU placement,
    DataLoader, and batching consistently with the training pipeline.

    Parameters
    ----------
    model        : Loaded DistilBertForSequenceClassification.
    tokenizer    : Matching DistilBertTokenizerFast.
    test_dataset : Tokenized Dataset with 'input_ids', 'attention_mask',
                   'labels' (integer 0–5). Must be the held-out test split.
    id2label     : Mapping from integer label to intent-name string.

    Returns
    -------
    EvalResult dataclass.
    """
    # Build minimal TrainingArguments for inference only.
    # output_dir is required but nothing is written there during predict().
    eval_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        # 'no_cuda' was removed in transformers 5.x; use 'use_cpu' instead.
        use_cpu=not torch.cuda.is_available(),
        fp16=False,   # predict() doesn't need mixed precision
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=get_data_collator(tokenizer),
        processing_class=tokenizer,
    )

    prediction_output = trainer.predict(test_dataset)
    logits  = prediction_output.predictions          # shape: (N, num_classes)
    y_true  = prediction_output.label_ids.astype(int)
    y_pred  = np.argmax(logits, axis=-1).astype(int)

    # Build label name list ordered by ID (0 → name, 1 → name, …)
    num_classes = logits.shape[1]
    label_names = [id2label[i] for i in range(num_classes)]

    accuracy  = float(accuracy_score(y_true, y_pred))
    macro_f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    rep_dict  = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    conf_mat  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
        accuracy=accuracy,
        macro_f1=macro_f1,
        report_dict=rep_dict,
        conf_matrix=conf_mat,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_report(result: EvalResult) -> str:
    """
    Format an EvalResult as a human-readable string.

    Includes overall metrics, per-class precision/recall/F1, and a
    confusion matrix with intent names on both axes.

    Parameters
    ----------
    result : EvalResult from evaluate_on_test().

    Returns
    -------
    Multi-line string ready for print().
    """
    lines: list[str] = []
    sep = "=" * 64
    lines.append(sep)
    lines.append("  TEST-SET EVALUATION RESULTS")
    lines.append(sep)

    # Overall
    lines.append(f"\n  Overall accuracy : {result.accuracy:.4f}  ({result.accuracy*100:.2f} %)")
    lines.append(f"  Macro F1         : {result.macro_f1:.4f}")
    lines.append(f"  Test examples    : {len(result.y_true)}")

    # Per-class table
    lines.append(f"\n  {'Intent':<46} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    lines.append(f"  {'-'*46} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for name in result.label_names:
        row = result.report_dict.get(name, {})
        p   = row.get("precision", 0.0)
        r   = row.get("recall",    0.0)
        f   = row.get("f1-score",  0.0)
        n   = int(row.get("support", 0))
        lines.append(f"  {name:<46} {p:>6.3f} {r:>6.3f} {f:>6.3f} {n:>5}")

    # Confusion matrix
    lines.append(f"\n  Confusion matrix  (rows = true, columns = predicted)")
    # Header: abbreviated labels (first 8 chars to keep alignment)
    abbrevs = [n[:12] for n in result.label_names]
    col_w = 14
    header = "  " + " " * 24 + "".join(f"{a:>{col_w}}" for a in abbrevs)
    lines.append(header)
    for i, row_name in enumerate(result.label_names):
        row_vals = "".join(f"{v:>{col_w}}" for v in result.conf_matrix[i])
        lines.append(f"  {row_name:<24}{row_vals}")

    lines.append(f"\n{sep}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(model_dir: str = OUTPUT_DIR) -> EvalResult:
    """
    Complete evaluation pipeline:
      1. Load model and tokenizer from model_dir.
      2. Load the test split via prepare_data().
      3. Tokenize the test split and rename 'label' → 'labels'.
      4. Run evaluate_on_test() once.
      5. Print the formatted report.

    The test split is used exactly once. This function must not be called
    during training, hyperparameter search, or checkpoint selection.

    Parameters
    ----------
    model_dir : Directory containing the saved best model (config.json +
                model.safetensors + tokenizer files). Defaults to OUTPUT_DIR.

    Returns
    -------
    EvalResult
    """
    sep = "=" * 64
    print(sep)
    print("  FINAL EVALUATION PIPELINE")
    print(sep)

    # -- 1. Load model ----------------------------------------------------------
    print(f"\n  [1/3] Loading model from '{model_dir}'...")
    model, tokenizer = load_saved_model(model_dir)
    print(f"  Model class  : {type(model).__name__}")
    print(f"  num_labels   : {model.config.num_labels}")
    print(f"  id2label     : {model.config.id2label}")

    # Build the id2label mapping with integer keys (config stores them as strings)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # Cross-check against the project's canonical mapping
    _, expected_id2label = build_label_mapping(SELECTED_INTENTS)
    assert id2label == expected_id2label, (
        f"Model id2label does not match config.SELECTED_INTENTS.\n"
        f"  model  : {id2label}\n"
        f"  config : {expected_id2label}"
    )
    print(f"  Label mapping verified against SELECTED_INTENTS ✓")

    # -- 2. Load test data ------------------------------------------------------
    print(f"\n  [2/3] Loading test split...")
    data = prepare_data()
    test_raw = data["test"]   # 240 examples, held out since Milestone 1
    print(f"  Test split   : {len(test_raw)} examples")
    print(f"  Classes      : {sorted(set(test_raw['label_text']))}")

    # Tokenize (no padding — handled by DataCollatorWithPadding in the Trainer)
    test_tok = tokenize_dataset(test_raw, tokenizer)
    # Rename 'label' → 'labels' to match model forward() signature
    test_tok = test_tok.rename_column("label", "labels")

    # -- 3. Evaluate ------------------------------------------------------------
    print(f"\n  [3/3] Running evaluation (test set, one pass)...")
    result = evaluate_on_test(model, tokenizer, test_tok, id2label)

    # Print full report
    print(format_report(result))

    return result


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
