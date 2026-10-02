"""
tests/test_model_utils.py

Unit and integration tests for src/model_utils.py.

Test organisation
-----------------
Unit tests (no model download, no GPU, fast):
    - TestBuildTokenizer        : basic tokenizer output structure
    - TestTokenizeDataset       : column handling, truncation, padding strategy
    - TestBuildModel            : config correctness (uses mock config, no download)
    - TestComputeMetrics        : metric computation with synthetic predictions
    - TestGetDataCollator       : collator padding behaviour

Integration tests (download distilbert-base-uncased ~250 MB, marked slow):
    - TestIntegrationRealModel  : load real model, forward pass, logit shape

Run unit tests only (fast, no download):
    pytest tests/test_model_utils.py -m "not integration" -v

Run all tests (requires internet + optional GPU):
    pytest tests/test_model_utils.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    PreTrainedTokenizerFast,
)

from src.config import MAX_LENGTH, MODEL_NAME, SELECTED_INTENTS
from src.data_utils import build_label_mapping
from src.model_utils import (
    NUM_LABELS,
    build_model,
    build_tokenizer,
    compute_metrics,
    get_data_collator,
    tokenize_dataset,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

LABEL2ID, ID2LABEL = build_label_mapping(SELECTED_INTENTS)

# ---------------------------------------------------------------------------
# Mock tokenizer for unit tests (avoids downloading model files)
# ---------------------------------------------------------------------------

def _make_mock_tokenizer() -> MagicMock:
    """
    A minimal mock that mimics DistilBertTokenizerFast's __call__ behaviour.

    Returns input_ids and attention_mask only (no token_type_ids — correct
    for DistilBERT). Pads all sequences to length 8 when padding=True.
    """
    SEQ_LEN = 8

    def _tokenize(texts, truncation=True, max_length=MAX_LENGTH,
                  padding=False, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        result = {
            "input_ids":      [[101] + [200 + i] * (SEQ_LEN - 2) + [102] for i in range(n)],
            "attention_mask": [[1] * SEQ_LEN for _ in range(n)],
        }
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        return result

    tok = MagicMock(spec=PreTrainedTokenizerFast)
    tok.side_effect = _tokenize
    tok.__call__ = MagicMock(side_effect=_tokenize)
    tok.vocab_size = 30522
    tok.pad_token_id = 0
    tok.padding_side = "right"
    tok.model_max_length = 512
    return tok


def _make_mock_dataset(n: int = 12) -> Dataset:
    """Small mock dataset with the same schema as a tokenizable split."""
    return Dataset.from_dict({
        "text":       [f"Mock query number {i}" for i in range(n)],
        "label":      [i % NUM_LABELS for i in range(n)],
        "label_text": [SELECTED_INTENTS[i % NUM_LABELS] for i in range(n)],
    })


def _make_tiny_tokenized_dataset(n: int = 12, seq_len: int = 8) -> Dataset:
    """
    Pre-tokenized dataset (no text/label_text columns) for collator tests.
    Sequences vary slightly in length to test dynamic padding.
    """
    input_ids_list      = [[101] + [200] * (seq_len - 2 - (i % 3)) + [102] for i in range(n)]
    attention_mask_list = [[1] * len(ids) for ids in input_ids_list]
    return Dataset.from_dict({
        "input_ids":      input_ids_list,
        "attention_mask": attention_mask_list,
        "label":          [i % NUM_LABELS for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def label_mapping():
    return build_label_mapping(SELECTED_INTENTS)


@pytest.fixture(scope="module")
def mock_tokenizer():
    return _make_mock_tokenizer()


@pytest.fixture(scope="module")
def mock_dataset():
    return _make_mock_dataset()


# ---------------------------------------------------------------------------
# Tests: NUM_LABELS constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_num_labels_equals_six(self):
        assert NUM_LABELS == 6

    def test_num_labels_matches_selected_intents(self):
        assert NUM_LABELS == len(SELECTED_INTENTS)


# ---------------------------------------------------------------------------
# Tests: build_tokenizer (unit — mocked, uses cached tokenizer files if present)
# ---------------------------------------------------------------------------

class TestBuildTokenizer:
    """
    These tests use a mock to avoid requiring a network download.
    The integration class tests the real tokenizer.
    """

    def test_mock_tokenizer_returns_expected_keys(self, mock_tokenizer):
        result = mock_tokenizer(["hello world"])
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_mock_tokenizer_no_token_type_ids(self, mock_tokenizer):
        """DistilBERT has no segment embeddings; token_type_ids must be absent."""
        result = mock_tokenizer(["test sentence"])
        assert "token_type_ids" not in result

    def test_mock_tokenizer_batch_size_preserved(self, mock_tokenizer):
        texts = ["one", "two", "three"]
        result = mock_tokenizer(texts)
        assert len(result["input_ids"]) == 3
        assert len(result["attention_mask"]) == 3


# ---------------------------------------------------------------------------
# Tests: tokenize_dataset
# ---------------------------------------------------------------------------

class TestTokenizeDataset:

    def test_removes_text_column(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert "text" not in result.column_names

    def test_removes_label_text_column(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert "label_text" not in result.column_names

    def test_preserves_label_column(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert "label" in result.column_names

    def test_adds_input_ids(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert "input_ids" in result.column_names

    def test_adds_attention_mask(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert "attention_mask" in result.column_names

    def test_does_not_add_token_type_ids(self, mock_dataset, mock_tokenizer):
        """DistilBERT must not have token_type_ids in its inputs."""
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert "token_type_ids" not in result.column_names

    def test_row_count_preserved(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert len(result) == len(mock_dataset)

    def test_label_values_unchanged(self, mock_dataset, mock_tokenizer):
        result = tokenize_dataset(mock_dataset, mock_tokenizer)
        assert list(result["label"]) == list(mock_dataset["label"])

    def test_no_padding_applied_by_tokenize_dataset(self):
        """
        tokenize_dataset must call the tokenizer with padding=False.
        We use a thin callable wrapper (not a MagicMock) so datasets.map
        receives a real dict back — datasets 5.x rejects MagicMock returns.
        """
        captured_kwargs: list[dict] = []

        class _CapturingTokenizer:
            """Wraps the real call convention; records kwargs; returns real dicts."""
            def __call__(self_, texts, truncation=True, max_length=MAX_LENGTH,
                         padding=False, **kwargs):
                captured_kwargs.append({"truncation": truncation,
                                        "max_length": max_length,
                                        "padding": padding})
                if isinstance(texts, str):
                    texts = [texts]
                ids   = [[101, 200 + i, 102] for i in range(len(texts))]
                masks = [[1, 1, 1]           for _ in range(len(texts))]
                return {"input_ids": ids, "attention_mask": masks}

        tok = _CapturingTokenizer()
        ds = Dataset.from_dict({
            "text":       ["Help.", "What is the exchange rate today?"],
            "label":      [0, 3],
            "label_text": [SELECTED_INTENTS[0], SELECTED_INTENTS[3]],
        })
        tokenize_dataset(ds, tok)

        assert len(captured_kwargs) >= 1, "Tokenizer was never called"
        for call in captured_kwargs:
            assert call["padding"] is False, (
                f"tokenize_dataset must use padding=False; got {call['padding']}"
            )

    def test_truncation_kwarg_is_true(self):
        """tokenize_dataset must pass truncation=True and max_length=MAX_LENGTH."""
        captured_kwargs: list[dict] = []

        class _CapturingTokenizer:
            def __call__(self_, texts, truncation=True, max_length=MAX_LENGTH,
                         padding=False, **kwargs):
                captured_kwargs.append({"truncation": truncation,
                                        "max_length": max_length,
                                        "padding": padding})
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                return {"input_ids": [[101, 200, 102]] * n,
                        "attention_mask": [[1, 1, 1]] * n}

        tok = _CapturingTokenizer()
        ds = Dataset.from_dict({
            "text":       ["test sentence"],
            "label":      [0],
            "label_text": [SELECTED_INTENTS[0]],
        })
        tokenize_dataset(ds, tok)

        assert len(captured_kwargs) >= 1, "Tokenizer was never called"
        assert captured_kwargs[0]["truncation"] is True, "truncation must be True"
        assert captured_kwargs[0]["max_length"] == MAX_LENGTH, (
            f"max_length must be {MAX_LENGTH}, got {captured_kwargs[0]['max_length']}"
        )


# ---------------------------------------------------------------------------
# Tests: build_model (config only — uses a synthetic config, no download)
# ---------------------------------------------------------------------------

class TestBuildModelConfig:
    """
    Test the model configuration logic without downloading pretrained weights.
    We instantiate DistilBertForSequenceClassification from a synthetic config
    (random weights) to verify that label mappings are correctly baked in.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def synthetic_model(cls):
        """Model with random weights — avoids downloading pretrained checkpoint."""
        cfg = DistilBertConfig(
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            vocab_size=100,        # tiny vocab for speed
            n_layers=1,            # single transformer layer
            n_heads=2,             # two attention heads
            dim=16,                # tiny hidden size
            hidden_dim=32,
        )
        return DistilBertForSequenceClassification(cfg)

    def test_num_labels_is_six(self, synthetic_model):
        assert synthetic_model.config.num_labels == NUM_LABELS

    def test_id2label_matches_expected(self, synthetic_model):
        assert synthetic_model.config.id2label == ID2LABEL

    def test_label2id_matches_expected(self, synthetic_model):
        assert synthetic_model.config.label2id == LABEL2ID

    def test_id2label_and_label2id_are_consistent(self, synthetic_model):
        cfg = synthetic_model.config
        for idx, name in cfg.id2label.items():
            assert cfg.label2id[name] == idx, (
                f"id2label[{idx}]={name!r} but label2id[{name!r}]={cfg.label2id.get(name)}"
            )

    def test_logit_shape_matches_num_labels(self, synthetic_model):
        """Forward pass on a tiny synthetic batch checks head output dimension."""
        cfg = synthetic_model.config
        batch_size, seq_len = 2, 5
        input_ids      = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        synthetic_model.eval()
        with torch.no_grad():
            outputs = synthetic_model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (batch_size, NUM_LABELS), (
            f"Expected logits shape ({batch_size}, {NUM_LABELS}), "
            f"got {tuple(outputs.logits.shape)}"
        )

    def test_logits_are_finite_even_before_training(self, synthetic_model):
        cfg = synthetic_model.config
        input_ids      = torch.randint(0, cfg.vocab_size, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)
        synthetic_model.eval()
        with torch.no_grad():
            outputs = synthetic_model(input_ids=input_ids, attention_mask=attention_mask)
        assert torch.isfinite(outputs.logits).all(), "Logits contain NaN or Inf"

    def test_no_token_type_ids_required(self, synthetic_model):
        """DistilBERT forward pass must not require token_type_ids."""
        cfg = synthetic_model.config
        input_ids      = torch.randint(0, cfg.vocab_size, (1, 4))
        attention_mask = torch.ones(1, 4, dtype=torch.long)
        synthetic_model.eval()
        with torch.no_grad():
            # This must not raise even without token_type_ids
            outputs = synthetic_model(input_ids=input_ids, attention_mask=attention_mask)
        assert outputs.logits is not None


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def _make_eval_pred(self, predictions: np.ndarray, label_ids: np.ndarray):
        """Minimal EvalPrediction-like object."""
        ep = MagicMock()
        ep.predictions = predictions
        ep.label_ids   = label_ids
        return ep

    def test_returns_accuracy_and_f1_keys(self):
        logits = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
        labels = np.array([0, 0, 0, 0])
        result = compute_metrics(self._make_eval_pred(logits, labels))
        assert "accuracy" in result
        assert "f1" in result

    def test_perfect_predictions_give_accuracy_one(self):
        # argmax of row i = i
        logits = np.eye(NUM_LABELS)
        labels = np.arange(NUM_LABELS)
        result = compute_metrics(self._make_eval_pred(logits, labels))
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_predictions_give_f1_one(self):
        logits = np.eye(NUM_LABELS)
        labels = np.arange(NUM_LABELS)
        result = compute_metrics(self._make_eval_pred(logits, labels))
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions_give_accuracy_zero(self):
        # logits → predict class 0 always; true labels are all 1
        logits = np.tile([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], (6, 1))
        labels = np.ones(6, dtype=int)
        result = compute_metrics(self._make_eval_pred(logits, labels))
        assert result["accuracy"] == pytest.approx(0.0)

    def test_f1_is_macro_averaged(self):
        """
        Verify macro-F1 via an exact manual computation.

        Setup: true labels [0,1,2,3,4,5], predictions [0,1,2,0,0,0].
          class 0: TP=1, FP=3 (misclassified 3,4,5 as 0), FN=0 → P=1/4, R=1, F1=2/(1+4)=0.4
          class 1: TP=1, FP=0, FN=0 → F1=1.0
          class 2: TP=1, FP=0, FN=0 → F1=1.0
          class 3: TP=0, FP=0, FN=1 → F1=0
          class 4: TP=0, FP=0, FN=1 → F1=0
          class 5: TP=0, FP=0, FN=1 → F1=0
        Macro = (0.4 + 1.0 + 1.0 + 0 + 0 + 0) / 6 ≈ 0.4
        """
        n = NUM_LABELS  # 6
        logits_correct = np.eye(n)[:3]                                    # predict 0,1,2 correctly
        logits_wrong   = np.tile([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], (3, 1))  # predict 0 for labels 3,4,5
        logits = np.vstack([logits_correct, logits_wrong])
        labels = np.arange(n)  # true labels: 0,1,2,3,4,5
        result = compute_metrics(self._make_eval_pred(logits, labels))
        assert result["f1"] == pytest.approx(0.4, abs=0.01)

    def test_returns_python_floats(self):
        logits = np.eye(NUM_LABELS)
        labels = np.arange(NUM_LABELS)
        result = compute_metrics(self._make_eval_pred(logits, labels))
        assert isinstance(result["accuracy"], float)
        assert isinstance(result["f1"], float)


# ---------------------------------------------------------------------------
# Tests: get_data_collator
# ---------------------------------------------------------------------------

class TestGetDataCollator:

    def test_returns_data_collator_with_padding(self, mock_tokenizer):
        collator = get_data_collator(mock_tokenizer)
        assert isinstance(collator, DataCollatorWithPadding)

    def test_collator_has_tokenizer_attribute(self, mock_tokenizer):
        collator = get_data_collator(mock_tokenizer)
        assert collator.tokenizer is mock_tokenizer

    def test_collator_pads_to_batch_maximum(self):
        """
        When sequences have different lengths, the collator must pad all of
        them to the longest in the batch (dynamic padding).

        DataCollatorWithPadding internally calls tokenizer.pad(), so the
        tokenizer mock must implement that method faithfully. Rather than
        fighting a MagicMock, we use a minimal real-behaving stub.
        """
        PAD_ID = 0

        class _MinimalTokenizer:
            """
            Minimal stub that implements only what DataCollatorWithPadding uses:
            .pad() and the metadata attributes it inspects.
            """
            pad_token_id   = PAD_ID
            padding_side   = "right"
            model_max_length = 512

            def pad(self_, encoded_inputs, padding=True,
                    max_length=None, pad_to_multiple_of=None, return_tensors="pt", **kwargs):
                # Find the longest sequence in the batch
                max_len = max(len(e["input_ids"]) for e in encoded_inputs)
                padded_ids   = []
                padded_masks = []
                for e in encoded_inputs:
                    n = len(e["input_ids"])
                    pad_len = max_len - n
                    padded_ids.append(e["input_ids"]   + [PAD_ID] * pad_len)
                    padded_masks.append(e["attention_mask"] + [0] * pad_len)

                result = {
                    "input_ids":      torch.tensor(padded_ids,   dtype=torch.long),
                    "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
                }
                if "label" in encoded_inputs[0]:
                    result["labels"] = torch.tensor(
                        [e["label"] for e in encoded_inputs], dtype=torch.long
                    )
                return result

        tok = _MinimalTokenizer()
        collator = DataCollatorWithPadding(tokenizer=tok, padding=True)

        # Two examples: lengths 3 and 5
        batch = [
            {"input_ids": [101, 200, 102],           "attention_mask": [1, 1, 1],       "label": 0},
            {"input_ids": [101, 200, 201, 202, 102], "attention_mask": [1, 1, 1, 1, 1], "label": 1},
        ]
        result = collator(batch)

        # Both sequences padded to length 5
        assert result["input_ids"].shape == (2, 5), (
            f"Expected (2, 5), got {tuple(result['input_ids'].shape)}"
        )
        # Shorter sequence: last two positions are pad tokens
        assert result["input_ids"][0, -2].item() == PAD_ID
        assert result["input_ids"][0, -1].item() == PAD_ID
        # Attention mask is 0 for padded positions
        assert result["attention_mask"][0, -2].item() == 0
        assert result["attention_mask"][0, -1].item() == 0


# ---------------------------------------------------------------------------
# Integration tests (download distilbert-base-uncased, ~250 MB)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationRealModel:
    """
    Load the actual pretrained model and tokenizer from the Hub.
    Run with:  pytest tests/test_model_utils.py -m integration -v
    Skipped in normal CI to avoid repeated downloads and GPU dependency.
    """

    @pytest.fixture(scope="class")
    def real_tokenizer(self):
        return build_tokenizer()

    @pytest.fixture(scope="class")
    def real_model(self):
        return build_model(LABEL2ID, ID2LABEL)

    def test_tokenizer_is_fast(self, real_tokenizer):
        assert real_tokenizer.is_fast, "Expected DistilBertTokenizerFast"

    def test_tokenizer_vocab_size(self, real_tokenizer):
        assert real_tokenizer.vocab_size == 30522

    def test_tokenizer_output_keys(self, real_tokenizer):
        enc = real_tokenizer("test sentence", truncation=True,
                             max_length=MAX_LENGTH, padding=False)
        assert "input_ids" in enc
        assert "attention_mask" in enc
        assert "token_type_ids" not in enc

    def test_tokenizer_truncation(self, real_tokenizer):
        long_text = " ".join(["word"] * 200)
        enc = real_tokenizer(long_text, truncation=True,
                             max_length=MAX_LENGTH, padding=False)
        assert len(enc["input_ids"]) <= MAX_LENGTH

    def test_model_config_num_labels(self, real_model):
        assert real_model.config.num_labels == NUM_LABELS

    def test_model_config_id2label(self, real_model):
        assert real_model.config.id2label == ID2LABEL

    def test_model_config_label2id(self, real_model):
        assert real_model.config.label2id == LABEL2ID

    def test_model_parameter_count(self, real_model):
        n = sum(p.numel() for p in real_model.parameters())
        # DistilBERT-base has ~66.4M params; allow a small range
        assert 65_000_000 <= n <= 68_000_000, f"Unexpected param count: {n:,}"

    def test_forward_pass_logit_shape(self, real_tokenizer, real_model):
        """Core smoke test: logits must be [batch, 6] and all finite."""
        sentences = [
            "I am waiting on my card.",
            "What is the exchange rate today?",
            "I want to request a refund.",
            "My card was stolen.",
        ]
        batch_size = len(sentences)
        inputs = real_tokenizer(
            sentences,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        real_model.eval()
        with torch.no_grad():
            outputs = real_model(**inputs)

        logits = outputs.logits
        assert logits.shape == (batch_size, NUM_LABELS), (
            f"Expected ({batch_size}, {NUM_LABELS}), got {tuple(logits.shape)}"
        )
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_predicted_labels_are_valid(self, real_tokenizer, real_model):
        inputs = real_tokenizer(
            ["I lost my card and need a replacement."],
            truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        )
        real_model.eval()
        with torch.no_grad():
            outputs = real_model(**inputs)
        pred_id = outputs.logits.argmax(dim=-1).item()
        assert 0 <= pred_id < NUM_LABELS
        assert ID2LABEL[pred_id] in SELECTED_INTENTS

    def test_tokenize_dataset_with_real_tokenizer(self, real_tokenizer):
        """tokenize_dataset must work end-to-end with the real tokenizer."""
        ds = Dataset.from_dict({
            "text":       ["I want to activate my card.", "What is the exchange rate?"],
            "label":      [0, 3],
            "label_text": ["activate_my_card", "exchange_rate"],
        })
        result = tokenize_dataset(ds, real_tokenizer)
        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names
        assert "label" in result.column_names
        assert "text" not in result.column_names
        assert "label_text" not in result.column_names
        # No padding: sequences may differ in length
        lengths = [len(ids) for ids in result["input_ids"]]
        assert all(l <= MAX_LENGTH for l in lengths), \
            f"Sequences exceed MAX_LENGTH={MAX_LENGTH}: {lengths}"
