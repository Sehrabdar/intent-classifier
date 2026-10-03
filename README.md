# Fine-Tuned Intent Classification System

An educational project demonstrating end-to-end Transformer fine-tuning using the
[Banking77](https://huggingface.co/datasets/mteb/banking77) dataset. The goal is
to learn real fine-tuning practice — not to build a production application.

**This project is CLI-based. No frontend, API server, database, or hosted LLM is used.**

---

## Overview

`distilbert-base-uncased` is fine-tuned on a six-class subset of Banking77 intents
using the Hugging Face Trainer API. The pipeline covers data preparation, training,
evaluation, and a command-line inference interface — each milestone implemented and
tested before the next begins.

---

## Completed Milestones

| Milestone | Status | Description |
|---|---|---|
| 1 | ✅ Complete | Project scaffolding, dataset preparation, leakage detection, tests |
| 2 | ✅ Complete | DistilBERT tokenization, model loading, data collation, tests |
| 3 | ✅ Complete | Training pipeline with Hugging Face Trainer, FP16, checkpoint selection |
| 4 | ✅ Complete | Final evaluation on held-out test set, confusion matrix, per-class metrics |
| 5 | ✅ Complete | CLI inference interface — single prediction and interactive mode |

---

## Supported Intents

Six Banking77 intents were selected for semantic diversity across card management,
security, balance, and currency domains. Labels are assigned alphabetically (0–5)
and are consistent across all data splits.

| Label ID | Intent | Description |
|---|---|---|
| 0 | `activate_my_card` | Activating a newly received card |
| 1 | `balance_not_updated_after_bank_transfer` | Balance not reflecting a recent transfer |
| 2 | `card_arrival` | Tracking or enquiring about a card delivery |
| 3 | `exchange_rate` | Asking about currency exchange rates |
| 4 | `lost_or_stolen_card` | Reporting a lost or stolen card |
| 5 | `request_refund` | Requesting a refund for a transaction |

---

## Model and Dataset

| Item | Detail |
|---|---|
| Base model | [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) |
| Parameters | 66,958,086 |
| Dataset | [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) (Parquet format) |
| Training config | 3 epochs · LR 2e-5 · batch 8 · warmup ratio 0.1 · weight decay 0.01 |
| Checkpoint selection | Lowest validation loss (`eval_strategy="epoch"`) |
| Mixed precision | FP16 (NVIDIA GTX 1650, 3.63 GB VRAM) |

> **Dataset note:** `PolyAI/banking77` cannot be loaded with `datasets>=2.x`
> (legacy script-based format). This project uses `mteb/banking77` — the same
> data in Parquet format.

---

## Final Test Results

Evaluated once on the held-out test split (240 examples, 40 per class, never seen during training or validation).

| Metric | Value |
|---|---|
| **Overall accuracy** | **97.50 %** |
| **Macro F1** | **0.9750** |
| Test examples | 240 |

### Per-class breakdown

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| activate_my_card | 1.000 | 0.950 | 0.974 | 40 |
| balance_not_updated_after_bank_transfer | 1.000 | 1.000 | 1.000 | 40 |
| card_arrival | 0.929 | 0.975 | 0.951 | 40 |
| exchange_rate | 1.000 | 1.000 | 1.000 | 40 |
| lost_or_stolen_card | 0.949 | 0.925 | 0.937 | 40 |
| request_refund | 0.976 | 1.000 | 0.988 | 40 |

> **Important:** These results apply only to the six-class Banking77 subset used
> in this project. They do not represent performance on all 77 Banking77 intents,
> on other datasets, or in any production environment.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.18.0 |
| datasets | 5.0.1 |
| accelerate | 1.15.0 |
| scikit-learn | 1.9.1 |
| GPU | NVIDIA GeForce GTX 1650 (3.63 GB VRAM) |

---

## Project Structure

```
intent-classifier/
├── src/
│   ├── __init__.py
│   ├── config.py          # All constants: dataset, model, hyperparameters, paths
│   ├── data_utils.py      # Load, verify, filter, split, leakage detection
│   ├── model_utils.py     # Tokenizer, model, metrics, data collator
│   ├── train.py           # Trainer setup, training loop, checkpoint management
│   ├── evaluate.py        # Final evaluation on held-out test set
│   └── predict.py         # CLI inference: single prediction and interactive mode
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_model_utils.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_predict.py
├── scripts/
│   └── trainer_smoke_test.py   # Standalone Trainer smoke test (development only)
├── outputs/               # Saved model + tokenizer (gitignored)
│   └── intent-classifier/ # Best checkpoint: model.safetensors, config.json, tokenizer.*
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Sehrabdar/intent-classifier.git
cd intent-classifier

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA (adjust for your CUDA version)
#    See: https://pytorch.org/get-started/locally/
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 4. Install remaining dependencies
pip install -r requirements.txt
```

For CPU-only use, install the standard PyTorch release (`pip install torch`) and
training will fall back to CPU automatically (slower, no FP16).

---

## Usage

### Verify the data pipeline

Loads `mteb/banking77`, verifies the six intents, builds splits, and checks for leakage:

```bash
python -m src.data_utils
```

### Train the model

Runs a 2-step smoke test first, then fine-tunes for 3 epochs. The best checkpoint
(by validation loss) is saved to `outputs/intent-classifier/`.

```bash
python -m src.train
```

Training takes approximately 70 seconds on an NVIDIA GTX 1650.

### Evaluate on the test set

Loads the saved model and evaluates the held-out 240-example test split **once**.
Do not run this repeatedly — the test set must not be used for tuning.

```bash
python -m src.evaluate
```

### Predict an intent

**Single prediction:**

```bash
python -m src.predict --text "My card has not arrived yet"
```

Example output:

```
  Loading model from 'outputs/intent-classifier'... ready.
  Device      : cuda
  num_labels  : 6
--------------------------------------------------------
  Input     : My card has not arrived yet
  Predicted : card_arrival
  Confidence: 91.8 %

  All class probabilities (highest first):
    card_arrival                                    91.8 % ██████████████████
    balance_not_updated_after_bank_transfer          2.2 %
    lost_or_stolen_card                              2.1 %
    activate_my_card                                 1.8 %
    request_refund                                   1.1 %
    exchange_rate                                    0.9 %
--------------------------------------------------------
```

**Interactive mode** (type queries one at a time; `exit` or Ctrl-D to quit):

```bash
python -m src.predict
```

```
  Intent Classifier — interactive mode
  Type a banking query and press Enter.
  Type 'exit' or 'quit' to stop, or press Ctrl-D.

  > What is the GBP to EUR exchange rate?
  Predicted : exchange_rate   Confidence: 94.7 %
  ...

  > exit
  Goodbye.
```

**Custom model directory:**

```bash
python -m src.predict --model-dir /path/to/other/checkpoint --text "I need a refund"
```

---

## Tests

```bash
# Fast unit tests — no download, no GPU required (~20 seconds)
pytest tests/ -m "not integration" -v

# Integration tests — download mteb/banking77 (~400 KB) and distilbert (~250 MB)
pytest tests/ -v

# Single module
pytest tests/test_predict.py -m "not integration" -v
pytest tests/test_evaluate.py -m "not integration" -v
pytest tests/test_train.py -m "not integration" -v
pytest tests/test_data_utils.py -v
```

Current unit-test count: **171 passed** across five test modules.

---

## Confidence Score and Limitations

### What the confidence score means

The confidence percentage shown by `src.predict` is the **softmax probability**
of the top class over the model's six raw logit scores. It is a measure of how
decisive the model is among the six options — not a calibrated real-world accuracy
estimate.

A 95% confidence does not mean the model is correct 95% of the time on arbitrary
text. It means the model assigns 95% of its internal probability mass to one class
out of six.

### Current limitations

| Limitation | Detail |
|---|---|
| **Six intents only** | The model recognises exactly six Banking77 intents. Any other banking query will be force-mapped to one of the six. |
| **Dataset-specific** | Trained and evaluated on Banking77 text. Performance on different phrasings or other domains is unknown. |
| **No calibration** | Softmax confidence is not calibrated. High confidence does not guarantee a correct prediction. |
| **No API or frontend** | This is a CLI-only research project. There is no web server, REST API, or web interface. |
| **No persistent storage** | Predictions are not logged or stored. |
| **No production deployment** | The model is not containerised, served, or monitored. It is not intended for production use. |

---

## Key Design Decisions

- **Label mapping built from strings, never from numeric IDs.** The `label_text` column
  is the source of truth; original Banking77 integer codes are discarded and replaced
  by a consistent 0–5 alphabetical mapping.
- **Stratified splitting.** `train_test_split(stratify_by_column="label")` ensures
  proportional class representation in train and validation.
- **Test set held out entirely.** The 240-example test split is never passed to the
  Trainer, never used for validation, and evaluated exactly once at the end.
- **Dynamic padding.** `DataCollatorWithPadding` pads per-batch rather than to
  `MAX_LENGTH` globally, saving VRAM on short Banking77 queries (avg ~10 tokens).
- **`eval_strategy = save_strategy = "epoch"`** satisfies the
  `load_best_model_at_end=True` constraint in transformers 5.18.0.
- **`warmup_ratio` does not exist in transformers 5.18.0.** `train.py` converts the
  ratio to `warmup_steps = ceil(ratio × total_steps)` at runtime.

---

## Reference

- Dataset: [mteb/banking77](https://huggingface.co/datasets/mteb/banking77)
- Model: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- Paper: [Efficient Intent Detection with Dual Sentence Encoders (Casanueva et al., 2020)](https://arxiv.org/abs/2003.04807)
