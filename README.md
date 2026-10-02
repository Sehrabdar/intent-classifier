# Fine-Tuned Intent Classification System

An educational project for learning Transformer fine-tuning using the
[Banking77](https://huggingface.co/datasets/mteb/banking77) dataset.

---

## Goal

Fine-tune `distilbert-base-uncased` on a 6-class subset of Banking77 intents
using the Hugging Face Trainer API. The result is a standalone CLI-based
classifier that can be saved, loaded, and evaluated.

**This is a learning project — no frontend, API server, or hosted LLM is used.**

---

## Selected Intents

Six Banking77 intents were chosen for semantic diversity:

| Label ID | Intent |
|---|---|
| 0 | `activate_my_card` |
| 1 | `balance_not_updated_after_bank_transfer` |
| 2 | `card_arrival` |
| 3 | `exchange_rate` |
| 4 | `lost_or_stolen_card` |
| 5 | `request_refund` |

Labels are assigned alphabetically (0–5) and are consistent across all splits.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.18.0 |
| datasets | 5.0.1 |
| accelerate | 1.15.0 |
| GPU | NVIDIA GeForce GTX 1650 (3.63 GB VRAM) |

---

## Project Structure

```
intent-classifier/
├── src/
│   ├── config.py          # All constants: dataset, model, hyperparameters
│   ├── data_utils.py      # Load, verify, filter, split, leakage detection
│   ├── model_utils.py     # Build model, compute_metrics  [Milestone 2]
│   ├── train.py           # Trainer setup and training loop  [Milestone 2]
│   ├── evaluate.py        # Full evaluation on test set  [Milestone 3]
│   └── predict.py         # Single-text inference  [Milestone 3]
├── tests/
│   ├── test_data_utils.py
│   ├── test_model_utils.py   [Milestone 2]
│   └── test_predict.py       [Milestone 3]
├── outputs/               # Saved model + tokenizer (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset Notes

> **Important:** `PolyAI/banking77` cannot be loaded with `datasets>=2.x`
> (legacy script-based dataset). This project uses `mteb/banking77` instead —
> the same data in Parquet format.

The dataset has **9,993 train** and **3,076 test** examples across 77 intents.
After filtering to 6 intents:

| Split | Size | Notes |
|---|---|---|
| Filtered train | 846 | Source for train/val split |
| Train | ~677 | 80% stratified split, seed 42 |
| Validation | ~169 | 20% stratified split, seed 42 |
| Test | 240 | Original test split, held out until final evaluation |

Zero text overlap exists between train/validation and the test split
(verified at runtime by `verify_no_leakage`).

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch with CUDA (adjust for your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### Milestone 1 — Data Preparation

Verify the dataset, build splits, and check for leakage:

```bash
python -m src.data_utils
```

### Milestone 2 — Training *(coming soon)*

```bash
python -m src.train
```

### Milestone 3 — Evaluation and Inference *(coming soon)*

```bash
python -m src.evaluate
python -m src.predict "I haven't received my card yet"
```

---

## Tests

```bash
# Fast unit tests only (no download required)
pytest tests/ -m "not integration" -v

# All tests including integration (downloads mteb/banking77 ~400 KB)
pytest tests/ -v
```

---

## Milestone Status

| Milestone | Status | Description |
|---|---|---|
| 1 | ✅ Complete | Scaffolding, dataset preparation, tests |
| 2 | ⬜ Pending | Model, training, checkpoint management |
| 3 | ⬜ Pending | Evaluation, confusion matrix, inference CLI |

---

## Key Design Decisions

- **Label mapping is built from strings, never from numeric IDs.** The `label_text`
  column is the source of truth; original Banking77 integer codes are discarded
  and replaced by a consistent 0–5 mapping.
- **Stratified splitting preserves class proportions.** `train_test_split` with
  `stratify_by_column="label"` ensures each class is proportionally represented
  in both train and validation.
- **Test set is never touched until final evaluation.** All hyperparameter
  decisions use the validation split only.
- **`save_strategy` and `eval_strategy` are both `"epoch"`** to satisfy the
  `load_best_model_at_end=True` constraint in transformers 5.18.0.

---

## Reference

- Dataset: [mteb/banking77](https://huggingface.co/datasets/mteb/banking77)
- Model: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- Paper: [Efficient Intent Detection with Dual Sentence Encoders (Casanueva et al., 2020)](https://arxiv.org/abs/2003.04807)
