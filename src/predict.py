"""
src/predict.py — CLI inference interface for the fine-tuned intent classifier.

Loads the saved model and tokenizer from OUTPUT_DIR once, then predicts the
intent of user-supplied text. Supports a one-shot flag and interactive mode.

Usage
-----
Single prediction:
    python -m src.predict --text "My card has not arrived yet"

Interactive mode (prompts until you type 'exit' or 'quit', or press Ctrl-D):
    python -m src.predict

Public API
----------
    predict_one(text, model, tokenizer, id2label, device) -> PredictResult
    format_prediction(result) -> str
    run_cli() -> None

Confidence note
---------------
The confidence score is derived from a softmax over the model's six raw logits.
Softmax produces values that sum to 1 and look like probabilities, but they are
NOT calibrated real-world certainties. A 99% softmax score means the model is
internally consistent — not that it is correct 99% of the time on new text.
This model is trained on a specific Banking77 subset; it may confidently produce
wrong answers on out-of-distribution input.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import MAX_LENGTH, OUTPUT_DIR, SELECTED_INTENTS
from src.evaluate import load_saved_model


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PredictResult:
    """
    Holds the output of a single prediction.

    Attributes
    ----------
    text            : The original input string.
    predicted_label : Canonical intent name (e.g. 'card_arrival').
    predicted_id    : Integer label ID (0–5).
    confidence      : Softmax probability of the top class (0–1).
    all_probs       : List of (label_name, probability) sorted high-to-low.
    """
    text:            str
    predicted_label: str
    predicted_id:    int
    confidence:      float
    all_probs:       list[tuple[str, float]]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_text(text: str) -> str:
    """
    Validate and clean a user-supplied input string.

    Raises
    ------
    ValueError : if the text is empty or whitespace-only after stripping.

    Returns
    -------
    The stripped text (leading/trailing whitespace removed).
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError(
            "Input text must not be empty or whitespace-only. "
            "Please provide a banking-related query."
        )
    return stripped


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_one(
    text: str,
    model,
    tokenizer,
    id2label: dict[int, str],
    device: torch.device,
) -> PredictResult:
    """
    Predict the intent for a single text string.

    Parameters
    ----------
    text      : Validated, non-empty input string.
    model     : Loaded DistilBertForSequenceClassification in eval mode.
    tokenizer : Matching DistilBertTokenizerFast.
    id2label  : Mapping from integer label ID to intent name.
    device    : torch.device to run inference on.

    Returns
    -------
    PredictResult with predicted intent, confidence, and all class probabilities.

    Implementation notes
    --------------------
    - truncation=True, max_length=MAX_LENGTH  : same as the training pipeline.
    - padding=True, return_tensors="pt"       : single-example batch.
    - torch.inference_mode()                  : disables gradient tracking for
                                                inference; lighter than no_grad.
    - model.eval()                            : disables dropout and batch-norm
                                                update (must be called before
                                                inference, not just at load time).
    """
    # Tokenize — same settings used in the training tokenization pipeline
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits[0]                    # shape: (num_classes,)
    probs  = F.softmax(logits, dim=-1)            # sum to 1.0

    pred_id    = int(probs.argmax().item())
    confidence = float(probs[pred_id].item())

    # Build sorted list (highest probability first)
    all_probs = sorted(
        [(id2label[i], float(probs[i].item())) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True,
    )

    return PredictResult(
        text=text,
        predicted_label=id2label[pred_id],
        predicted_id=pred_id,
        confidence=confidence,
        all_probs=all_probs,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_prediction(result: PredictResult) -> str:
    """
    Format a PredictResult as a human-readable block of text.

    Parameters
    ----------
    result : PredictResult from predict_one().

    Returns
    -------
    Multi-line string ready for print().
    """
    lines: list[str] = []
    sep = "-" * 56
    lines.append(sep)
    lines.append(f"  Input     : {result.text}")
    lines.append(f"  Predicted : {result.predicted_label}")
    lines.append(f"  Confidence: {result.confidence * 100:.1f} %")
    lines.append(f"")
    lines.append(f"  All class probabilities (highest first):")
    for label, prob in result.all_probs:
        bar   = "█" * int(prob * 20)          # 20-char max bar
        lines.append(f"    {label:<46} {prob * 100:>5.1f} % {bar}")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loading with actionable error messages
# ---------------------------------------------------------------------------

def _load_model_checked(model_dir: str):
    """
    Load model and tokenizer from model_dir, with clear error messages.

    Checks for the directory and key files before attempting to load, so
    users get an actionable message rather than a raw FileNotFoundError.

    Returns
    -------
    (model, tokenizer, id2label, device)
    """
    # Check directory exists
    if not os.path.isdir(model_dir):
        print(
            f"  ERROR: Model directory not found: '{model_dir}'\n"
            f"  Run training first:\n"
            f"    python -m src.train",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check essential files are present
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    missing  = [f for f in required if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        print(
            f"  ERROR: Missing files in '{model_dir}': {missing}\n"
            f"  The model directory may be incomplete. Re-run training:\n"
            f"    python -m src.train",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Loading model from '{model_dir}'...", end=" ", flush=True)
    try:
        model, tokenizer = load_saved_model(model_dir)
    except Exception as exc:
        print("FAILED", file=sys.stderr)
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("ready.")

    # Build id2label with integer keys (config.json stores keys as strings)
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()   # set once at load time; predict_one() sets it again defensively

    print(f"  Device      : {device}")
    print(f"  num_labels  : {model.config.num_labels}")
    return model, tokenizer, id2label, device


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_cli() -> None:
    """
    Parse CLI arguments and dispatch to single-shot or interactive mode.

    Single-shot:
        python -m src.predict --text "My card has not arrived yet"

    Interactive:
        python -m src.predict
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Predict a banking intent from a text query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m src.predict --text "My card has not arrived yet"\n'
            "  python -m src.predict          # interactive mode\n\n"
            "Supported intents:\n"
            + "\n".join(f"  - {intent}" for intent in SELECTED_INTENTS)
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to classify. Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Path to saved model directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    # Load model once (reused for every prediction in interactive mode)
    model, tokenizer, id2label, device = _load_model_checked(args.model_dir)

    if args.text is not None:
        # ── Single-shot mode ────────────────────────────────────────────────
        try:
            clean = validate_text(args.text)
        except ValueError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        result = predict_one(clean, model, tokenizer, id2label, device)
        print(format_prediction(result))

    else:
        # ── Interactive mode ────────────────────────────────────────────────
        print("\n  Intent Classifier — interactive mode")
        print("  Type a banking query and press Enter.")
        print("  Type 'exit' or 'quit' to stop, or press Ctrl-D.\n")

        while True:
            try:
                raw = input("  > ")
            except EOFError:
                # Ctrl-D / end of piped input — exit cleanly
                print("\n  Goodbye.")
                break

            if raw.strip().lower() in {"exit", "quit", "q"}:
                print("  Goodbye.")
                break

            try:
                clean = validate_text(raw)
            except ValueError as exc:
                print(f"  {exc}\n")
                continue

            result = predict_one(clean, model, tokenizer, id2label, device)
            print(format_prediction(result))
            print()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cli()
