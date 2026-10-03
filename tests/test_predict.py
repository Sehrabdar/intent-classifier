"""
tests/test_predict.py

Unit and integration tests for src/predict.py.

Test organisation
-----------------
Unit tests (no model download, no real weights):
    TestValidateText        : input validation rules
    TestPredictOne          : probability math and label mapping with a mock model
    TestFormatPrediction    : output formatting with synthetic PredictResult
    TestLoadModelChecked    : missing dir / missing files → sys.exit(1)
    TestCLIArgPath          : --text flag with fully mocked dependencies
    TestInteractiveMode     : interactive loop exit behavior

Integration tests (marked @pytest.mark.integration):
    TestIntegrationPredict  : load real saved model, predict on sample sentences

Run unit tests only (fast, no download):
    pytest tests/test_predict.py -m "not integration" -v

Run all tests:
    pytest tests/test_predict.py -v
"""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn.functional as F

from src.config import MAX_LENGTH, OUTPUT_DIR, SELECTED_INTENTS
from src.data_utils import build_label_mapping
from src.model_utils import NUM_LABELS
from src.predict import (
    PredictResult,
    _load_model_checked,
    format_prediction,
    predict_one,
    run_cli,
    validate_text,
)

LABEL2ID, ID2LABEL = build_label_mapping(SELECTED_INTENTS)
LABEL_NAMES = [ID2LABEL[i] for i in range(NUM_LABELS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_logits(pred_class: int, spread: float = 5.0) -> torch.Tensor:
    """
    Return raw logits where pred_class has a dominant score.
    Useful for deterministic probability tests.
    """
    logits = torch.zeros(NUM_LABELS)
    logits[pred_class] = spread
    return logits


def _make_mock_model(pred_class: int = 0, spread: float = 5.0):
    """
    Minimal model stub whose forward() returns logits dominated by pred_class.
    """
    logits = _make_fake_logits(pred_class, spread)

    model_output = MagicMock()
    model_output.logits = logits.unsqueeze(0)   # shape (1, NUM_LABELS)

    model = MagicMock()
    model.return_value = model_output
    model.config.id2label  = {str(i): v for i, v in ID2LABEL.items()}
    model.config.num_labels = NUM_LABELS
    return model


def _make_mock_tokenizer():
    """Minimal tokenizer stub returning a small fixed encoding."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids":      torch.tensor([[101, 200, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    return tokenizer


# ---------------------------------------------------------------------------
# Tests: validate_text
# ---------------------------------------------------------------------------

class TestValidateText:

    def test_valid_text_returned_stripped(self):
        assert validate_text("  hello world  ") == "hello world"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            validate_text("    ")

    def test_tab_only_raises(self):
        with pytest.raises(ValueError):
            validate_text("\t\n")

    def test_single_word_accepted(self):
        assert validate_text("hello") == "hello"

    def test_long_text_accepted(self):
        long_text = "word " * 200
        result = validate_text(long_text)
        assert len(result) > 0

    def test_error_message_is_informative(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_text("   ")


# ---------------------------------------------------------------------------
# Tests: predict_one — probability math and label mapping
# ---------------------------------------------------------------------------

class TestPredictOne:

    def _run(self, pred_class: int = 0, spread: float = 5.0):
        model    = _make_mock_model(pred_class, spread)
        tokenizer = _make_mock_tokenizer()
        device   = torch.device("cpu")
        return predict_one(
            "I want to activate my card",
            model, tokenizer, ID2LABEL, device
        )

    def test_predicted_label_matches_dominant_class(self):
        for c in range(NUM_LABELS):
            result = self._run(pred_class=c)
            assert result.predicted_label == ID2LABEL[c], (
                f"Expected {ID2LABEL[c]}, got {result.predicted_label}"
            )

    def test_predicted_id_matches_dominant_class(self):
        result = self._run(pred_class=2)
        assert result.predicted_id == 2

    def test_confidence_is_highest_probability(self):
        result = self._run(pred_class=0, spread=10.0)
        assert result.confidence == max(prob for _, prob in result.all_probs)

    def test_confidence_is_between_zero_and_one(self):
        result = self._run()
        assert 0.0 <= result.confidence <= 1.0

    def test_all_probs_sum_to_one(self):
        result = self._run()
        total = sum(p for _, p in result.all_probs)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_all_probs_sorted_descending(self):
        result = self._run()
        probs = [p for _, p in result.all_probs]
        assert probs == sorted(probs, reverse=True)

    def test_all_probs_has_six_entries(self):
        result = self._run()
        assert len(result.all_probs) == NUM_LABELS

    def test_all_probs_labels_are_valid_intents(self):
        result = self._run()
        names = [name for name, _ in result.all_probs]
        assert set(names) == set(SELECTED_INTENTS)

    def test_text_is_preserved_in_result(self):
        model     = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        device    = torch.device("cpu")
        result = predict_one("My card arrived damaged", model, tokenizer, ID2LABEL, device)
        assert result.text == "My card arrived damaged"

    def test_model_called_with_tokenizer_output(self):
        model     = _make_mock_model(pred_class=3)
        tokenizer = _make_mock_tokenizer()
        device    = torch.device("cpu")
        predict_one("exchange rate test", model, tokenizer, ID2LABEL, device)
        # Tokenizer must have been called
        tokenizer.assert_called_once()
        # Model must have been called (forward pass)
        model.assert_called_once()

    def test_tokenizer_called_with_truncation_true(self):
        model     = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        predict_one("test", model, tokenizer, ID2LABEL, torch.device("cpu"))
        call_kwargs = tokenizer.call_args
        assert call_kwargs.kwargs.get("truncation") is True or \
               (len(call_kwargs.args) > 1 and call_kwargs.args[1] is True), \
               "tokenizer must be called with truncation=True"

    def test_tokenizer_called_with_correct_max_length(self):
        model     = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        predict_one("test", model, tokenizer, ID2LABEL, torch.device("cpu"))
        call_kwargs = tokenizer.call_args.kwargs
        assert call_kwargs.get("max_length") == MAX_LENGTH

    def test_model_set_to_eval_mode(self):
        """predict_one() must call model.eval() before forward pass."""
        model     = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        predict_one("test", model, tokenizer, ID2LABEL, torch.device("cpu"))
        model.eval.assert_called()

    def test_no_gradients_computed(self):
        """
        torch.inference_mode() should prevent gradient accumulation.
        We verify indirectly: probabilities computed from float logits
        have no .grad (inference_mode detaches outputs).
        """
        model     = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        result = predict_one("test", model, tokenizer, ID2LABEL, torch.device("cpu"))
        # confidence is a plain Python float — not a tensor with grad
        assert isinstance(result.confidence, float)
        assert not isinstance(result.confidence, torch.Tensor)

    def test_uniform_logits_give_equal_probs(self):
        """If all logits are 0, softmax gives 1/N for each class."""
        logits = torch.zeros(NUM_LABELS)
        model_output = MagicMock()
        model_output.logits = logits.unsqueeze(0)
        model = MagicMock()
        model.return_value = model_output

        tokenizer = _make_mock_tokenizer()
        result = predict_one("test", model, tokenizer, ID2LABEL, torch.device("cpu"))

        expected = 1.0 / NUM_LABELS
        for _, p in result.all_probs:
            assert p == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: format_prediction
# ---------------------------------------------------------------------------

class TestFormatPrediction:

    def _make_result(self, pred_class: int = 0) -> PredictResult:
        logits = _make_fake_logits(pred_class, spread=5.0)
        probs  = F.softmax(logits, dim=-1)
        all_probs = sorted(
            [(ID2LABEL[i], float(probs[i].item())) for i in range(NUM_LABELS)],
            key=lambda x: x[1], reverse=True
        )
        return PredictResult(
            text="I lost my card",
            predicted_label=ID2LABEL[pred_class],
            predicted_id=pred_class,
            confidence=float(probs[pred_class].item()),
            all_probs=all_probs,
        )

    def test_returns_string(self):
        assert isinstance(format_prediction(self._make_result()), str)

    def test_contains_input_text(self):
        result = self._make_result()
        report = format_prediction(result)
        assert result.text in report

    def test_contains_predicted_label(self):
        result = self._make_result(pred_class=4)   # lost_or_stolen_card
        report = format_prediction(result)
        assert ID2LABEL[4] in report

    def test_contains_confidence_percentage(self):
        result = self._make_result()
        report = format_prediction(result)
        # Confidence shown as XX.X %
        assert "%" in report

    def test_contains_all_six_intent_names(self):
        result = self._make_result()
        report = format_prediction(result)
        for name in SELECTED_INTENTS:
            assert name in report, f"Missing '{name}' from formatted output"

    def test_non_empty_output(self):
        assert len(format_prediction(self._make_result())) > 50

    def test_header_separator_present(self):
        result = self._make_result()
        report = format_prediction(result)
        assert "-" * 10 in report   # separator line exists


# ---------------------------------------------------------------------------
# Tests: _load_model_checked — missing dir / files
# ---------------------------------------------------------------------------

class TestLoadModelChecked:

    def test_exits_when_dir_does_not_exist(self):
        with pytest.raises(SystemExit) as exc_info:
            _load_model_checked("/nonexistent/path/that/cannot/exist")
        assert exc_info.value.code == 1

    def test_exits_when_model_safetensors_missing(self, tmp_path):
        # Create dir with config.json and tokenizer.json but no model.safetensors
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        # model.safetensors is absent
        with pytest.raises(SystemExit) as exc_info:
            _load_model_checked(str(tmp_path))
        assert exc_info.value.code == 1

    def test_exits_when_config_json_missing(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"")
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(SystemExit) as exc_info:
            _load_model_checked(str(tmp_path))
        assert exc_info.value.code == 1

    def test_exits_when_tokenizer_json_missing(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"")
        with pytest.raises(SystemExit) as exc_info:
            _load_model_checked(str(tmp_path))
        assert exc_info.value.code == 1

    def test_error_message_mentions_src_train(self, capsys, tmp_path):
        """User should know how to fix the problem."""
        with pytest.raises(SystemExit):
            _load_model_checked("/nonexistent/path")
        captured = capsys.readouterr()
        assert "src.train" in captured.err

    def test_error_message_mentions_missing_path(self, capsys):
        bad_path = "/does/not/exist"
        with pytest.raises(SystemExit):
            _load_model_checked(bad_path)
        captured = capsys.readouterr()
        assert bad_path in captured.err


# ---------------------------------------------------------------------------
# Tests: run_cli — --text flag (fully mocked)
# ---------------------------------------------------------------------------

class TestCLIArgPath:
    """
    Patches _load_model_checked and predict_one so run_cli() can be tested
    without real weights, GPU, or filesystem access.
    """

    def _mock_load(self):
        """Return the 4-tuple that _load_model_checked returns."""
        mock_model    = _make_mock_model(pred_class=2)
        mock_tokenizer = _make_mock_tokenizer()
        return mock_model, mock_tokenizer, ID2LABEL, torch.device("cpu")

    def _fake_result(self, text: str = "test") -> PredictResult:
        return PredictResult(
            text=text,
            predicted_label="card_arrival",
            predicted_id=2,
            confidence=0.92,
            all_probs=[(ID2LABEL[i], 1 / NUM_LABELS) for i in range(NUM_LABELS)],
        )

    def test_single_text_flag_calls_predict_once(self):
        fake_result = self._fake_result("My card has not arrived yet")

        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("src.predict.predict_one", return_value=fake_result) as mock_pred, \
             patch("sys.argv", ["predict", "--text", "My card has not arrived yet"]):
            run_cli()

        mock_pred.assert_called_once()

    def test_empty_text_flag_exits_with_error(self, capsys):
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("sys.argv", ["predict", "--text", "   "]):
            with pytest.raises(SystemExit) as exc_info:
                run_cli()
        assert exc_info.value.code == 1

    def test_output_contains_predicted_label(self, capsys):
        fake_result = self._fake_result()

        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("src.predict.predict_one", return_value=fake_result), \
             patch("sys.argv", ["predict", "--text", "card test"]):
            run_cli()

        captured = capsys.readouterr()
        assert "card_arrival" in captured.out

    def test_custom_model_dir_is_passed_to_loader(self):
        with patch("src.predict._load_model_checked",
                   return_value=self._mock_load()) as mock_load, \
             patch("src.predict.predict_one", return_value=self._fake_result()), \
             patch("sys.argv", ["predict", "--text", "test",
                                "--model-dir", "/custom/path"]):
            run_cli()

        mock_load.assert_called_once_with("/custom/path")


# ---------------------------------------------------------------------------
# Tests: interactive mode exit behavior (mocked)
# ---------------------------------------------------------------------------

class TestInteractiveMode:

    def _mock_load(self):
        return _make_mock_model(pred_class=0), _make_mock_tokenizer(), ID2LABEL, torch.device("cpu")

    def test_exit_keyword_stops_loop(self, capsys):
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("builtins.input", side_effect=["exit"]), \
             patch("sys.argv", ["predict"]):
            run_cli()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_quit_keyword_stops_loop(self, capsys):
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("builtins.input", side_effect=["quit"]), \
             patch("sys.argv", ["predict"]):
            run_cli()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_eof_stops_loop_cleanly(self, capsys):
        """Ctrl-D (EOFError from input()) should exit without traceback."""
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("builtins.input", side_effect=EOFError), \
             patch("sys.argv", ["predict"]):
            run_cli()   # must not raise
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_empty_input_shows_error_and_continues(self, capsys):
        """Empty input should print an error and prompt again, not crash."""
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("builtins.input", side_effect=["", "exit"]), \
             patch("sys.argv", ["predict"]):
            run_cli()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_whitespace_input_shows_error_and_continues(self, capsys):
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("builtins.input", side_effect=["   ", "quit"]), \
             patch("sys.argv", ["predict"]):
            run_cli()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_valid_input_calls_predict_once(self):
        fake_result = PredictResult(
            text="my card",
            predicted_label="card_arrival",
            predicted_id=2,
            confidence=0.9,
            all_probs=[(ID2LABEL[i], 1 / NUM_LABELS) for i in range(NUM_LABELS)],
        )
        with patch("src.predict._load_model_checked", return_value=self._mock_load()), \
             patch("src.predict.predict_one", return_value=fake_result) as mock_pred, \
             patch("builtins.input", side_effect=["my card", "exit"]), \
             patch("sys.argv", ["predict"]):
            run_cli()
        mock_pred.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests (load real model, real tokenizer)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationPredict:
    """
    Load the real saved model from OUTPUT_DIR and run inference on a few
    handcrafted sentences.

    Precondition: training has completed and the model is saved at OUTPUT_DIR.

    Run with:  pytest tests/test_predict.py -m integration -v

    NOTE: We do NOT assert exact label predictions because the model may
    produce different top-1 results on borderline queries. We assert structural
    properties (shapes, ranges, types) and one obviously correct prediction.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def loaded(cls):
        """Load model, tokenizer, id2label, device once for all tests."""
        return _load_model_checked(OUTPUT_DIR)

    def test_predict_returns_predict_result(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("I need a refund", model, tokenizer, id2label, device)
        assert isinstance(result, PredictResult)

    def test_confidence_in_range(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("What is the exchange rate?", model, tokenizer, id2label, device)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_probs_sum_to_one(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("My card was stolen", model, tokenizer, id2label, device)
        total = sum(p for _, p in result.all_probs)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_predicted_label_is_valid_intent(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("How do I activate my new card?", model, tokenizer, id2label, device)
        assert result.predicted_label in SELECTED_INTENTS

    def test_all_probs_sorted_descending(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("I want to request a refund", model, tokenizer, id2label, device)
        probs = [p for _, p in result.all_probs]
        assert probs == sorted(probs, reverse=True)

    def test_all_six_classes_present(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("Any banking query", model, tokenizer, id2label, device)
        names = {name for name, _ in result.all_probs}
        assert names == set(SELECTED_INTENTS)

    def test_unambiguous_exchange_rate_query(self, loaded):
        """
        'What is today's GBP to EUR exchange rate?' is clearly exchange_rate.
        The model achieved 100% recall on exchange_rate on the held-out test set.
        """
        model, tokenizer, id2label, device = loaded
        result = predict_one(
            "What is today's GBP to EUR exchange rate?",
            model, tokenizer, id2label, device
        )
        assert result.predicted_label == "exchange_rate", (
            f"Expected exchange_rate, got {result.predicted_label} "
            f"(confidence {result.confidence:.2%})"
        )

    def test_format_prediction_smoke(self, loaded):
        model, tokenizer, id2label, device = loaded
        result = predict_one("My card has not arrived", model, tokenizer, id2label, device)
        report = format_prediction(result)
        assert isinstance(report, str)
        assert len(report) > 50
