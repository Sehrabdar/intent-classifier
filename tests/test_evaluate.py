"""
tests/test_evaluate.py

Unit tests for src/evaluate.py.

Test organisation
-----------------
Unit tests (no model download, no GPU, fast):
    TestEvalResult         : dataclass construction and field access
    TestFormatReport       : report formatting from synthetic EvalResult
    TestEvaluateOnTest     : metric computation via mocked Trainer.predict()
    TestRunEvaluationMocks : orchestration function with all I/O mocked

Integration tests (marked @pytest.mark.integration):
    TestIntegrationEvaluate: load real saved model, run on real test split

Run unit tests only:
    pytest tests/test_evaluate.py -m "not integration" -v

Run all tests:
    pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset

from src.config import SELECTED_INTENTS, OUTPUT_DIR
from src.data_utils import build_label_mapping
from src.evaluate import (
    EvalResult,
    evaluate_on_test,
    format_report,
    load_saved_model,
    run_evaluation,
)
from src.model_utils import NUM_LABELS

LABEL2ID, ID2LABEL = build_label_mapping(SELECTED_INTENTS)
LABEL_NAMES = [ID2LABEL[i] for i in range(NUM_LABELS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect_result() -> EvalResult:
    """EvalResult where every prediction is correct."""
    n = 60  # 10 per class
    y_true = np.array([i % NUM_LABELS for i in range(n)], dtype=int)
    y_pred = y_true.copy()
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )
    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        label_names=LABEL_NAMES,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        report_dict=classification_report(
            y_true, y_pred,
            labels=list(range(NUM_LABELS)),
            target_names=LABEL_NAMES,
            output_dict=True,
        ),
        conf_matrix=confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS))),
    )


def _make_random_result(seed: int = 0) -> EvalResult:
    """EvalResult with random (mostly wrong) predictions."""
    rng = np.random.default_rng(seed)
    n = 60
    y_true = np.array([i % NUM_LABELS for i in range(n)], dtype=int)
    y_pred = rng.integers(0, NUM_LABELS, size=n)
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )
    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        label_names=LABEL_NAMES,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        report_dict=classification_report(
            y_true, y_pred,
            labels=list(range(NUM_LABELS)),
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        conf_matrix=confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS))),
    )


def _make_mock_tokenized_dataset(n: int = 12) -> Dataset:
    """Tokenized dataset with 'labels' column (already renamed)."""
    return Dataset.from_dict({
        "input_ids":      [[101, 200, 102]] * n,
        "attention_mask": [[1, 1, 1]] * n,
        "labels":         [i % NUM_LABELS for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests: EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:

    def test_perfect_accuracy(self):
        r = _make_perfect_result()
        assert r.accuracy == pytest.approx(1.0)

    def test_perfect_macro_f1(self):
        r = _make_perfect_result()
        assert r.macro_f1 == pytest.approx(1.0)

    def test_label_names_length(self):
        r = _make_perfect_result()
        assert len(r.label_names) == NUM_LABELS

    def test_label_names_match_selected_intents(self):
        r = _make_perfect_result()
        assert r.label_names == LABEL_NAMES

    def test_conf_matrix_shape(self):
        r = _make_perfect_result()
        assert r.conf_matrix.shape == (NUM_LABELS, NUM_LABELS)

    def test_conf_matrix_diagonal_perfect(self):
        """Perfect predictions → all counts on the diagonal."""
        r = _make_perfect_result()
        off_diag = r.conf_matrix - np.diag(np.diag(r.conf_matrix))
        assert off_diag.sum() == 0

    def test_report_dict_has_all_intent_keys(self):
        r = _make_perfect_result()
        for name in LABEL_NAMES:
            assert name in r.report_dict, f"Missing class '{name}' in report_dict"

    def test_report_dict_has_macro_avg(self):
        r = _make_perfect_result()
        assert "macro avg" in r.report_dict

    def test_report_dict_precision_recall_f1_keys(self):
        r = _make_perfect_result()
        for name in LABEL_NAMES:
            row = r.report_dict[name]
            assert "precision" in row
            assert "recall" in row
            assert "f1-score" in row
            assert "support" in row

    def test_y_true_y_pred_lengths_match(self):
        r = _make_random_result()
        assert len(r.y_true) == len(r.y_pred)

    def test_accuracy_below_one_for_random_predictions(self):
        r = _make_random_result()
        assert r.accuracy < 1.0

    def test_accuracy_range(self):
        r = _make_random_result()
        assert 0.0 <= r.accuracy <= 1.0

    def test_macro_f1_range(self):
        r = _make_random_result()
        assert 0.0 <= r.macro_f1 <= 1.0


# ---------------------------------------------------------------------------
# Tests: format_report
# ---------------------------------------------------------------------------

class TestFormatReport:

    def test_returns_string(self):
        r = _make_perfect_result()
        assert isinstance(format_report(r), str)

    def test_contains_overall_accuracy(self):
        r = _make_perfect_result()
        report = format_report(r)
        assert "accuracy" in report.lower() or "Accuracy" in report or "100.00" in report

    def test_contains_macro_f1(self):
        r = _make_perfect_result()
        report = format_report(r)
        assert "macro" in report.lower() or "F1" in report

    def test_contains_all_intent_names(self):
        r = _make_perfect_result()
        report = format_report(r)
        for name in LABEL_NAMES:
            # Names may be truncated in the confusion matrix; check full name in per-class table
            assert name in report, f"Intent '{name}' missing from report"

    def test_contains_confusion_matrix_label(self):
        r = _make_perfect_result()
        report = format_report(r)
        assert "confusion" in report.lower() or "Confusion" in report

    def test_contains_test_count(self):
        r = _make_perfect_result()
        report = format_report(r)
        assert "60" in report  # 60 test examples

    def test_format_is_non_empty(self):
        r = _make_random_result()
        assert len(format_report(r)) > 100

    def test_precision_recall_f1_headers_present(self):
        r = _make_perfect_result()
        report = format_report(r)
        assert "Prec" in report or "prec" in report.lower()
        assert "Rec" in report  or "rec"  in report.lower()
        assert "F1"  in report  or "f1"   in report.lower()


# ---------------------------------------------------------------------------
# Tests: evaluate_on_test (Trainer.predict mocked)
# ---------------------------------------------------------------------------

class TestEvaluateOnTest:
    """
    Patches Trainer.predict() so that evaluate_on_test() can be tested
    without a real model, GPU, or data download.
    """

    def _make_mock_prediction_output(self, y_true, y_pred):
        """Build a PredictionOutput-like mock from y_true and y_pred arrays."""
        n_classes = NUM_LABELS
        # Build logits so that argmax gives y_pred
        logits = np.zeros((len(y_pred), n_classes), dtype=np.float32)
        for i, p in enumerate(y_pred):
            logits[i, p] = 10.0   # high logit for predicted class

        mock_output = MagicMock()
        mock_output.predictions = logits
        mock_output.label_ids   = np.array(y_true, dtype=np.int64)
        return mock_output

    def test_perfect_accuracy_from_perfect_predictions(self):
        y = list(range(NUM_LABELS)) * 10   # 60 examples
        pred_output = self._make_mock_prediction_output(y, y)

        mock_model   = MagicMock()
        mock_tok     = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output
        mock_dataset = _make_mock_tokenized_dataset()

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            result = evaluate_on_test(mock_model, mock_tok, mock_dataset, ID2LABEL)

        assert result.accuracy == pytest.approx(1.0)
        assert result.macro_f1 == pytest.approx(1.0)

    def test_zero_accuracy_from_all_wrong_predictions(self):
        n = 60
        y_true = [i % NUM_LABELS for i in range(n)]
        # Shift all predictions by 1 (no overlap with true labels)
        y_pred = [(i + 1) % NUM_LABELS for i in range(n)]

        pred_output = self._make_mock_prediction_output(y_true, y_pred)
        mock_model   = MagicMock()
        mock_tok     = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output
        mock_dataset = _make_mock_tokenized_dataset()

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            result = evaluate_on_test(mock_model, mock_tok, mock_dataset, ID2LABEL)

        assert result.accuracy == pytest.approx(0.0)

    def test_label_names_ordered_by_id(self):
        y = [0] * 10
        pred_output = self._make_mock_prediction_output(y, y)
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            result = evaluate_on_test(MagicMock(), MagicMock(),
                                      _make_mock_tokenized_dataset(), ID2LABEL)

        assert result.label_names == [ID2LABEL[i] for i in range(NUM_LABELS)]

    def test_conf_matrix_shape(self):
        y = [i % NUM_LABELS for i in range(60)]
        pred_output = self._make_mock_prediction_output(y, y)
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            result = evaluate_on_test(MagicMock(), MagicMock(),
                                      _make_mock_tokenized_dataset(), ID2LABEL)

        assert result.conf_matrix.shape == (NUM_LABELS, NUM_LABELS)

    def test_report_dict_contains_all_classes(self):
        y = [i % NUM_LABELS for i in range(60)]
        pred_output = self._make_mock_prediction_output(y, y)
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            result = evaluate_on_test(MagicMock(), MagicMock(),
                                      _make_mock_tokenized_dataset(), ID2LABEL)

        for name in LABEL_NAMES:
            assert name in result.report_dict

    def test_trainer_predict_called_once(self):
        y = [0] * 12
        pred_output = self._make_mock_prediction_output(y, y)
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            evaluate_on_test(MagicMock(), MagicMock(),
                             _make_mock_tokenized_dataset(), ID2LABEL)

        mock_trainer.predict.assert_called_once()

    def test_y_true_matches_label_ids(self):
        y_true = [2, 3, 4, 5, 0, 1]
        y_pred = y_true[:]
        pred_output = self._make_mock_prediction_output(y_true, y_pred)
        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = pred_output

        with patch("src.evaluate.Trainer", return_value=mock_trainer):
            result = evaluate_on_test(MagicMock(), MagicMock(),
                                      _make_mock_tokenized_dataset(n=6), ID2LABEL)

        assert list(result.y_true) == y_true
        assert list(result.y_pred) == y_pred


# ---------------------------------------------------------------------------
# Tests: run_evaluation (fully mocked I/O)
# ---------------------------------------------------------------------------

class TestRunEvaluationMocked:
    """
    Patches load_saved_model, prepare_data, tokenize_dataset, and
    evaluate_on_test so that run_evaluation() can be tested without any
    real disk I/O, model download, or dataset download.
    """

    def _mock_model(self):
        model = MagicMock()
        # Simulate what model.config.id2label returns (string keys from JSON)
        model.config.id2label = {str(i): v for i, v in ID2LABEL.items()}
        model.config.num_labels = NUM_LABELS
        return model

    def _mock_prepare_data(self):
        n = 12
        test_ds = Dataset.from_dict({
            "text":       [f"text {i}" for i in range(n)],
            "label":      [i % NUM_LABELS for i in range(n)],
            "label_text": [SELECTED_INTENTS[i % NUM_LABELS] for i in range(n)],
        })
        return {
            "train":   Dataset.from_dict({"text": [], "label": [], "label_text": []}),
            "val":     Dataset.from_dict({"text": [], "label": [], "label_text": []}),
            "test":    test_ds,
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
        }

    def test_run_evaluation_returns_eval_result(self):
        fake_result = _make_perfect_result()

        with patch("src.evaluate.load_saved_model",
                   return_value=(self._mock_model(), MagicMock())), \
             patch("src.evaluate.prepare_data", return_value=self._mock_prepare_data()), \
             patch("src.evaluate.tokenize_dataset",
                   return_value=_make_mock_tokenized_dataset().rename_column("labels", "label")), \
             patch("src.evaluate.evaluate_on_test", return_value=fake_result):
            result = run_evaluation()

        assert isinstance(result, EvalResult)

    def test_evaluate_on_test_called_once(self):
        fake_result = _make_perfect_result()
        mock_eval = MagicMock(return_value=fake_result)

        with patch("src.evaluate.load_saved_model",
                   return_value=(self._mock_model(), MagicMock())), \
             patch("src.evaluate.prepare_data", return_value=self._mock_prepare_data()), \
             patch("src.evaluate.tokenize_dataset",
                   return_value=_make_mock_tokenized_dataset().rename_column("labels", "label")), \
             patch("src.evaluate.evaluate_on_test", mock_eval):
            run_evaluation()

        mock_eval.assert_called_once()

    def test_load_saved_model_called_with_output_dir(self):
        fake_result = _make_perfect_result()
        mock_load = MagicMock(return_value=(self._mock_model(), MagicMock()))

        with patch("src.evaluate.load_saved_model", mock_load), \
             patch("src.evaluate.prepare_data", return_value=self._mock_prepare_data()), \
             patch("src.evaluate.tokenize_dataset",
                   return_value=_make_mock_tokenized_dataset().rename_column("labels", "label")), \
             patch("src.evaluate.evaluate_on_test", return_value=fake_result):
            run_evaluation(model_dir=OUTPUT_DIR)

        mock_load.assert_called_once_with(OUTPUT_DIR)

    def test_label_mapping_verified_against_selected_intents(self):
        """
        If the saved model's id2label does not match SELECTED_INTENTS,
        run_evaluation() must raise AssertionError.
        """
        bad_model = MagicMock()
        bad_model.config.id2label = {"0": "wrong_intent", "1": "another_wrong"}
        bad_model.config.num_labels = NUM_LABELS

        with patch("src.evaluate.load_saved_model",
                   return_value=(bad_model, MagicMock())), \
             patch("src.evaluate.prepare_data", return_value=self._mock_prepare_data()), \
             patch("src.evaluate.tokenize_dataset",
                   return_value=_make_mock_tokenized_dataset().rename_column("labels", "label")):
            with pytest.raises(AssertionError, match="id2label"):
                run_evaluation()


# ---------------------------------------------------------------------------
# Integration tests (require saved model at OUTPUT_DIR)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationEvaluate:
    """
    Load the real saved model from OUTPUT_DIR and run one evaluation pass
    on the actual held-out test split.

    Precondition: training has completed and the model is saved at OUTPUT_DIR.

    Run with:  pytest tests/test_evaluate.py -m integration -v
    """

    @pytest.fixture(scope="class")
    @classmethod
    def eval_result(cls):
        return run_evaluation(model_dir=OUTPUT_DIR)

    def test_test_set_size_is_240(self, eval_result):
        assert len(eval_result.y_true) == 240

    def test_accuracy_above_training_epoch3(self, eval_result):
        """
        With val accuracy of 0.9647 at epoch 3, test accuracy should be
        in a reasonable range — not drastically lower.
        """
        assert eval_result.accuracy >= 0.70, (
            f"Test accuracy {eval_result.accuracy:.4f} is suspiciously low; "
            "check that the correct model checkpoint was loaded."
        )

    def test_macro_f1_is_positive(self, eval_result):
        assert eval_result.macro_f1 > 0.0

    def test_all_six_classes_in_report(self, eval_result):
        for name in LABEL_NAMES:
            assert name in eval_result.report_dict, f"Missing class '{name}' in report"

    def test_per_class_support_sums_to_240(self, eval_result):
        total = sum(
            int(eval_result.report_dict[name]["support"])
            for name in LABEL_NAMES
        )
        assert total == 240

    def test_each_class_has_40_support(self, eval_result):
        """Banking77 test set has exactly 40 examples per class."""
        for name in LABEL_NAMES:
            support = int(eval_result.report_dict[name]["support"])
            assert support == 40, f"{name}: expected support=40, got {support}"

    def test_conf_matrix_row_sums_to_40(self, eval_result):
        """Each true class has 40 test examples."""
        for i, row_sum in enumerate(eval_result.conf_matrix.sum(axis=1)):
            assert row_sum == 40, f"Row {i} ({LABEL_NAMES[i]}): sum={row_sum}"

    def test_label_names_match_selected_intents(self, eval_result):
        assert eval_result.label_names == LABEL_NAMES

    def test_accuracy_and_y_arrays_consistent(self, eval_result):
        correct = (eval_result.y_true == eval_result.y_pred).sum()
        computed = correct / len(eval_result.y_true)
        assert computed == pytest.approx(eval_result.accuracy, abs=1e-6)
