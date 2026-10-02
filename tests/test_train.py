"""
tests/test_train.py

Unit tests for src/train.py.

Test organisation
-----------------
Unit tests (no model download, no GPU, fast):
    TestBuildTrainingArgs   : validates argument computation from config
    TestRenameLabel         : the label → labels column rename
    TestSelectPerClass      : balanced subset selection helper
    TestRunSmokeMocked      : smoke test with Trainer replaced by a mock

Integration tests (marked @pytest.mark.integration):
    TestIntegrationSmoke    : real 2-step Trainer smoke test on 24 examples

Run fast unit tests:
    pytest tests/test_train.py -m "not integration" -v

Run all including integration:
    pytest tests/test_train.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from transformers import TrainingArguments

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
from src.data_utils import build_label_mapping
from src.model_utils import NUM_LABELS
from src.train import (
    _rename_label_column,
    _select_per_class,
    build_training_args,
    run_smoke_test,
)

LABEL2ID, ID2LABEL = build_label_mapping(SELECTED_INTENTS)

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_labeled_dataset(n: int = 24) -> Dataset:
    """
    Tiny dataset that matches the schema produced by filter_and_remap():
    columns: text, label (int 0–5), label_text (str).
    """
    return Dataset.from_dict({
        "text":       [f"Example sentence number {i}." for i in range(n)],
        "label":      [i % NUM_LABELS for i in range(n)],
        "label_text": [SELECTED_INTENTS[i % NUM_LABELS] for i in range(n)],
    })


def _make_tokenized_dataset(n: int = 24, seq_len: int = 8) -> Dataset:
    """
    Pre-tokenized dataset with 'label' column (not yet renamed to 'labels').
    Simulates the output of tokenize_dataset().
    """
    return Dataset.from_dict({
        "input_ids":      [[101] + [200] * (seq_len - 2) + [102]] * n,
        "attention_mask": [[1] * seq_len] * n,
        "label":          [i % NUM_LABELS for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests: build_training_args
# ---------------------------------------------------------------------------

class TestBuildTrainingArgs:

    def test_returns_training_arguments_instance(self):
        args = build_training_args(num_train_examples=100)
        assert isinstance(args, TrainingArguments)

    def test_num_train_epochs(self):
        args = build_training_args(num_train_examples=100)
        assert args.num_train_epochs == NUM_EPOCHS

    def test_per_device_train_batch_size(self):
        args = build_training_args(num_train_examples=100)
        assert args.per_device_train_batch_size == BATCH_SIZE

    def test_learning_rate(self):
        args = build_training_args(num_train_examples=100)
        assert args.learning_rate == pytest.approx(LEARNING_RATE)

    def test_weight_decay(self):
        args = build_training_args(num_train_examples=100)
        assert args.weight_decay == pytest.approx(WEIGHT_DECAY)

    def test_warmup_steps_computed_from_ratio(self):
        """
        transformers 5.18.0 has no warmup_ratio param.
        build_training_args() must compute warmup_steps = ceil(ratio * total).
        total_steps = ceil(100/8) * 3 = 13 * 3 = 39
        warmup_steps = ceil(0.1 * 39) = ceil(3.9) = 4
        """
        steps_per_epoch = math.ceil(100 / BATCH_SIZE)
        total_steps     = steps_per_epoch * NUM_EPOCHS
        expected        = math.ceil(WARMUP_RATIO * total_steps)
        args = build_training_args(num_train_examples=100)
        assert args.warmup_steps == expected

    def test_warmup_steps_scales_with_dataset_size(self):
        """Larger dataset → more total steps → more warmup steps."""
        args_small = build_training_args(num_train_examples=50)
        args_large = build_training_args(num_train_examples=1000)
        assert args_large.warmup_steps > args_small.warmup_steps

    def test_eval_strategy_is_epoch(self):
        args = build_training_args(num_train_examples=100)
        # TrainingArguments stores strategies as enum objects; == compares by value.
        assert args.eval_strategy == "epoch"

    def test_save_strategy_is_epoch(self):
        args = build_training_args(num_train_examples=100)
        # TrainingArguments stores strategies as enum objects; == compares by value.
        assert args.save_strategy == "epoch"

    def test_save_and_eval_strategy_match(self):
        """Strategies must match for load_best_model_at_end=True to work."""
        args = build_training_args(num_train_examples=100)
        assert args.eval_strategy == args.save_strategy

    def test_load_best_model_at_end(self):
        args = build_training_args(num_train_examples=100)
        assert args.load_best_model_at_end is True

    def test_metric_for_best_model(self):
        args = build_training_args(num_train_examples=100)
        assert args.metric_for_best_model == "eval_loss"

    def test_greater_is_better_false_for_loss(self):
        """eval_loss should be minimised, not maximised."""
        args = build_training_args(num_train_examples=100)
        assert args.greater_is_better is False

    def test_output_dir_default(self):
        args = build_training_args(num_train_examples=100)
        assert args.output_dir == OUTPUT_DIR

    def test_output_dir_custom(self):
        args = build_training_args(num_train_examples=100, output_dir="/tmp/custom")
        assert args.output_dir == "/tmp/custom"

    def test_fp16_disabled_when_no_cuda(self):
        """fp16 must be False when CUDA is unavailable, even if config says True."""
        with patch("src.train.torch.cuda.is_available", return_value=False):
            args = build_training_args(num_train_examples=100, fp16=True)
        assert args.fp16 is False

    def test_report_to_none(self):
        """No external logging services should be enabled by default."""
        args = build_training_args(num_train_examples=100)
        # report_to is stored as a list internally
        assert args.report_to == [] or args.report_to == ["none"]

    def test_seed(self):
        args = build_training_args(num_train_examples=100)
        assert args.seed == SEED


# ---------------------------------------------------------------------------
# Tests: _rename_label_column
# ---------------------------------------------------------------------------

class TestRenameLabel:

    def test_column_renamed_from_label_to_labels(self):
        ds = _make_tokenized_dataset(n=6)
        assert "label" in ds.column_names
        result = _rename_label_column(ds)
        assert "labels" in result.column_names
        assert "label" not in result.column_names

    def test_label_values_preserved(self):
        ds = _make_tokenized_dataset(n=6)
        result = _rename_label_column(ds)
        assert list(result["labels"]) == list(ds["label"])

    def test_other_columns_untouched(self):
        ds = _make_tokenized_dataset(n=6)
        result = _rename_label_column(ds)
        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names

    def test_row_count_unchanged(self):
        ds = _make_tokenized_dataset(n=12)
        result = _rename_label_column(ds)
        assert len(result) == 12


# ---------------------------------------------------------------------------
# Tests: _select_per_class
# ---------------------------------------------------------------------------

class TestSelectPerClass:

    def test_returns_correct_total_size(self):
        ds = _make_labeled_dataset(n=24)
        result = _select_per_class(ds, n_per_class=2)
        assert len(result) == 2 * NUM_LABELS

    def test_each_class_has_n_examples(self):
        ds = _make_labeled_dataset(n=24)
        n = 3
        result = _select_per_class(ds, n_per_class=n)
        from collections import Counter
        counts = Counter(result["label"])
        for label in range(NUM_LABELS):
            assert counts[label] == n, f"Class {label} has {counts[label]} examples, expected {n}"

    def test_reproducible_with_same_seed(self):
        ds = _make_labeled_dataset(n=24)
        r1 = _select_per_class(ds, n_per_class=2, seed=42)
        r2 = _select_per_class(ds, n_per_class=2, seed=42)
        assert list(r1["label"]) == list(r2["label"])
        assert list(r1["text"])  == list(r2["text"])

    def test_different_seeds_may_differ(self):
        """With a dataset large enough to have shuffleable choices, seeds differ."""
        ds = _make_labeled_dataset(n=60)   # 10 per class → shuffling has effect
        r1 = _select_per_class(ds, n_per_class=3, seed=1)
        r2 = _select_per_class(ds, n_per_class=3, seed=99)
        # At least one class should have different examples selected
        assert list(r1["text"]) != list(r2["text"])

    def test_n_per_class_one_returns_six_examples(self):
        ds = _make_labeled_dataset(n=24)
        result = _select_per_class(ds, n_per_class=1)
        assert len(result) == NUM_LABELS


# ---------------------------------------------------------------------------
# Tests: run_smoke_test (Trainer mocked — fast, no GPU, no download)
# ---------------------------------------------------------------------------

class TestRunSmokeMocked:
    """
    Patch the Trainer so run_smoke_test() can be tested without a real GPU
    or model download. Verifies that run_smoke_test() correctly calls train()
    and evaluate(), and that it raises on non-finite losses.
    """

    def _make_mock_trainer(self, train_loss=1.5, eval_loss=1.6, eval_acc=0.3):
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(
            metrics={"train_loss": train_loss},
            global_step=2,
        )
        trainer.evaluate.return_value = {
            "eval_loss":     eval_loss,
            "eval_accuracy": eval_acc,
            "eval_f1":       0.25,
            "eval_runtime":  0.1,
        }
        return trainer

    def _make_mock_tokenizer(self):
        """Minimal tokenizer stub compatible with run_smoke_test()."""
        class _CapturingTokenizer:
            is_fast = True
            vocab_size = 100
            pad_token_id = 0
            padding_side = "right"
            model_max_length = 512

            def __call__(self_, texts, truncation=True, max_length=64,
                         padding=False, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                return {"input_ids": [[101, 200, 102]] * n,
                        "attention_mask": [[1, 1, 1]] * n}

            def save_pretrained(self_, path):
                pass

        return _CapturingTokenizer()

    def test_smoke_test_calls_trainer_train(self):
        mock_tok = self._make_mock_tokenizer()
        mock_trainer = self._make_mock_trainer()

        with patch("src.train.Trainer", return_value=mock_trainer), \
             patch("src.train.build_model", return_value=MagicMock()), \
             patch("src.train.TrainingArguments", return_value=MagicMock()), \
             patch("src.train.shutil.rmtree"), \
             patch("src.train.os.path.exists", return_value=False):
            train_ds = _make_labeled_dataset(n=24)
            val_ds   = _make_labeled_dataset(n=12)
            run_smoke_test(train_ds, val_ds, LABEL2ID, ID2LABEL, mock_tok)

        mock_trainer.train.assert_called_once()

    def test_smoke_test_calls_evaluate(self):
        mock_tok = self._make_mock_tokenizer()
        mock_trainer = self._make_mock_trainer()

        with patch("src.train.Trainer", return_value=mock_trainer), \
             patch("src.train.build_model", return_value=MagicMock()), \
             patch("src.train.TrainingArguments", return_value=MagicMock()), \
             patch("src.train.shutil.rmtree"), \
             patch("src.train.os.path.exists", return_value=False):
            train_ds = _make_labeled_dataset(n=24)
            val_ds   = _make_labeled_dataset(n=12)
            run_smoke_test(train_ds, val_ds, LABEL2ID, ID2LABEL, mock_tok)

        mock_trainer.evaluate.assert_called_once()

    def test_smoke_test_raises_on_nan_train_loss(self):
        mock_tok = self._make_mock_tokenizer()
        mock_trainer = self._make_mock_trainer(train_loss=float("nan"))

        with patch("src.train.Trainer", return_value=mock_trainer), \
             patch("src.train.build_model", return_value=MagicMock()), \
             patch("src.train.TrainingArguments", return_value=MagicMock()), \
             patch("src.train.shutil.rmtree"), \
             patch("src.train.os.path.exists", return_value=False):
            train_ds = _make_labeled_dataset(n=24)
            val_ds   = _make_labeled_dataset(n=12)
            with pytest.raises(AssertionError, match="finite"):
                run_smoke_test(train_ds, val_ds, LABEL2ID, ID2LABEL, mock_tok)

    def test_smoke_test_raises_on_nan_eval_loss(self):
        mock_tok = self._make_mock_tokenizer()
        mock_trainer = self._make_mock_trainer(eval_loss=float("nan"))

        with patch("src.train.Trainer", return_value=mock_trainer), \
             patch("src.train.build_model", return_value=MagicMock()), \
             patch("src.train.TrainingArguments", return_value=MagicMock()), \
             patch("src.train.shutil.rmtree"), \
             patch("src.train.os.path.exists", return_value=False):
            train_ds = _make_labeled_dataset(n=24)
            val_ds   = _make_labeled_dataset(n=12)
            with pytest.raises(AssertionError, match="finite"):
                run_smoke_test(train_ds, val_ds, LABEL2ID, ID2LABEL, mock_tok)

    def test_smoke_test_passes_finite_losses(self):
        """Sanity check: run_smoke_test() must NOT raise with finite losses."""
        mock_tok = self._make_mock_tokenizer()
        mock_trainer = self._make_mock_trainer(train_loss=1.5, eval_loss=1.6)

        with patch("src.train.Trainer", return_value=mock_trainer), \
             patch("src.train.build_model", return_value=MagicMock()), \
             patch("src.train.TrainingArguments", return_value=MagicMock()), \
             patch("src.train.shutil.rmtree"), \
             patch("src.train.os.path.exists", return_value=False):
            train_ds = _make_labeled_dataset(n=24)
            val_ds   = _make_labeled_dataset(n=12)
            # Must not raise
            run_smoke_test(train_ds, val_ds, LABEL2ID, ID2LABEL, mock_tok)


# ---------------------------------------------------------------------------
# Integration smoke test (real Trainer, tiny subset, 2 steps)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationSmoke:
    """
    Runs run_smoke_test() with the real Trainer, tokenizer, and model.
    Downloads distilbert-base-uncased if not cached.
    Requires CUDA for fp16; falls back gracefully on CPU.

    Run with:  pytest tests/test_train.py -m integration -v
    """

    @pytest.fixture(scope="class")
    @classmethod
    def prepared_data(cls):
        from src.data_utils import prepare_data
        return prepare_data()

    @pytest.fixture(scope="class")
    @classmethod
    def real_tokenizer(cls):
        from src.model_utils import build_tokenizer
        return build_tokenizer()

    def test_smoke_does_not_raise(self, prepared_data, real_tokenizer):
        """The 2-step smoke test must complete without any error."""
        run_smoke_test(
            train_data=prepared_data["train"],
            val_data=prepared_data["val"],
            label2id=prepared_data["label2id"],
            id2label=prepared_data["id2label"],
            tokenizer=real_tokenizer,
        )

    def test_smoke_output_dir_created(self, prepared_data, real_tokenizer):
        """A checkpoint must be written to the smoke output dir."""
        import os
        from src.train import _SMOKE_OUTPUT_DIR
        run_smoke_test(
            train_data=prepared_data["train"],
            val_data=prepared_data["val"],
            label2id=prepared_data["label2id"],
            id2label=prepared_data["id2label"],
            tokenizer=real_tokenizer,
        )
        checkpoints = [d for d in os.listdir(_SMOKE_OUTPUT_DIR)
                       if d.startswith("checkpoint-")]
        assert checkpoints, f"No checkpoint found in {_SMOKE_OUTPUT_DIR}"
