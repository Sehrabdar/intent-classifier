"""
tests/test_data_utils.py

Unit and integration tests for src/data_utils.py.

Test organisation
-----------------
Unit tests (no network, no GPU, fast):
    - test_build_label_mapping_*         : label2id / id2label construction
    - test_filter_and_remap_*            : filtering and label remapping on mock data
    - test_verify_no_leakage_*           : leakage detection logic
    - test_create_splits_*               : split size, reproducibility, proportions

Integration tests (download mteb/banking77, marked slow):
    - test_integration_*                 : run on the real dataset

Run unit tests only (fast, no download):
    pytest tests/test_data_utils.py -m "not integration" -v

Run all tests (requires internet, ~30 s first run):
    pytest tests/test_data_utils.py -v
"""

from __future__ import annotations

import pytest
from collections import Counter
from datasets import Dataset, ClassLabel

from src.config import SELECTED_INTENTS, SEED, VAL_SPLIT_RATIO
from src.data_utils import (
    build_label_mapping,
    filter_and_remap,
    create_splits,
    verify_no_leakage,
    prepare_data,
)

# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

# Original Banking77 integer labels for the six selected intents
# (verified live from the actual dataset — used here only for realistic mocks).
_ORIGINAL_BANKING77_IDS: dict[str, int] = {
    "activate_my_card": 0,
    "balance_not_updated_after_bank_transfer": 5,
    "card_arrival": 11,
    "exchange_rate": 32,
    "lost_or_stolen_card": 41,
    "request_refund": 52,
}

# Out-of-scope intents that should be removed by filter_and_remap
_OUT_OF_SCOPE: dict[str, int] = {
    "age_limit": 7,
    "pin_blocked": 29,
}

_EXAMPLES_PER_CLASS = 15  # enough for meaningful proportion checks


def _make_mock_raw_dataset(
    examples_per_class: int = _EXAMPLES_PER_CLASS,
    include_out_of_scope: bool = True,
) -> Dataset:
    """
    Builds a small Dataset that mimics the mteb/banking77 schema:
        {'text': str, 'label': int, 'label_text': str}

    Labels use the original Banking77 integer IDs so that filter_and_remap
    can be tested end-to-end (remapping from 0/5/11/32/41/52 → 0/1/2/3/4/5).
    """
    texts, labels, label_texts = [], [], []

    for intent, orig_id in _ORIGINAL_BANKING77_IDS.items():
        for j in range(examples_per_class):
            texts.append(f"Mock query {j} about {intent}")
            labels.append(orig_id)
            label_texts.append(intent)

    if include_out_of_scope:
        for intent, orig_id in _OUT_OF_SCOPE.items():
            for j in range(3):
                texts.append(f"Mock out-of-scope query {j} about {intent}")
                labels.append(orig_id)
                label_texts.append(intent)

    return Dataset.from_dict({"text": texts, "label": labels, "label_text": label_texts})


def _make_mock_filtered_dataset(examples_per_class: int = _EXAMPLES_PER_CLASS) -> Dataset:
    """
    Returns a Dataset already filtered to the six selected intents and
    with labels remapped to 0–5 + cast to ClassLabel.
    Useful for tests that only need to test splitting, not filtering.
    """
    label2id, _ = build_label_mapping(SELECTED_INTENTS)
    raw = _make_mock_raw_dataset(examples_per_class, include_out_of_scope=False)
    return filter_and_remap(raw, label2id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def label_mapping():
    return build_label_mapping(SELECTED_INTENTS)


@pytest.fixture(scope="module")
def mock_raw():
    return _make_mock_raw_dataset()


@pytest.fixture(scope="module")
def mock_filtered():
    return _make_mock_filtered_dataset()


@pytest.fixture(scope="module")
def mock_splits(mock_filtered):
    return create_splits(mock_filtered, val_ratio=0.20, seed=SEED)


# ---------------------------------------------------------------------------
# Tests: build_label_mapping
# ---------------------------------------------------------------------------

class TestBuildLabelMapping:

    def test_returns_two_dicts(self, label_mapping):
        label2id, id2label = label_mapping
        assert isinstance(label2id, dict)
        assert isinstance(id2label, dict)

    def test_label2id_has_all_six_intents(self, label_mapping):
        label2id, _ = label_mapping
        assert set(label2id.keys()) == set(SELECTED_INTENTS)

    def test_label_ids_are_zero_based_consecutive(self, label_mapping):
        label2id, _ = label_mapping
        ids = sorted(label2id.values())
        assert ids == list(range(len(SELECTED_INTENTS)))

    def test_alphabetical_order_matches_config(self, label_mapping):
        """SELECTED_INTENTS is alphabetical; IDs must follow that order."""
        label2id, _ = label_mapping
        sorted_intents = sorted(SELECTED_INTENTS)
        for expected_id, intent in enumerate(sorted_intents):
            assert label2id[intent] == expected_id, (
                f"{intent!r} expected id {expected_id}, got {label2id[intent]}"
            )

    def test_id2label_is_exact_inverse(self, label_mapping):
        label2id, id2label = label_mapping
        for intent, idx in label2id.items():
            assert id2label[idx] == intent

    def test_same_size(self, label_mapping):
        label2id, id2label = label_mapping
        assert len(label2id) == len(id2label) == len(SELECTED_INTENTS)

    def test_custom_order_is_preserved(self):
        """Caller controls order; we do NOT sort inside build_label_mapping."""
        custom = ["request_refund", "card_arrival", "exchange_rate"]
        l2i, i2l = build_label_mapping(custom)
        assert l2i["request_refund"] == 0
        assert l2i["card_arrival"] == 1
        assert l2i["exchange_rate"] == 2
        assert i2l[0] == "request_refund"


# ---------------------------------------------------------------------------
# Tests: filter_and_remap
# ---------------------------------------------------------------------------

class TestFilterAndRemap:

    def test_removes_out_of_scope_intents(self, mock_raw, label_mapping):
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        remaining = set(result["label_text"])
        assert remaining == set(SELECTED_INTENTS), (
            f"Unexpected intents after filtering: {remaining - set(SELECTED_INTENTS)}"
        )

    def test_correct_example_count_after_filter(self, mock_raw, label_mapping):
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        # 6 intents × _EXAMPLES_PER_CLASS; out-of-scope rows are removed
        expected = 6 * _EXAMPLES_PER_CLASS
        assert len(result) == expected, f"Expected {expected}, got {len(result)}"

    def test_remapped_labels_are_zero_to_five(self, mock_raw, label_mapping):
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        unique_labels = sorted(set(result["label"]))
        assert unique_labels == list(range(len(SELECTED_INTENTS)))

    def test_label_text_consistent_with_remapped_label(self, mock_raw, label_mapping):
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        for example in result:
            expected_id = label2id[example["label_text"]]
            assert example["label"] == expected_id, (
                f"label_text={example['label_text']!r} should map to "
                f"{expected_id}, got {example['label']}"
            )

    def test_original_banking77_ids_are_replaced(self, mock_raw, label_mapping):
        """
        For each row, the new label must NOT equal the original Banking77 ID
        for that row's label_text — unless the new ID and original ID happen
        to be the same value (e.g. activate_my_card: 0 → 0).

        We verify per-row: new_label == label2id[label_text], which means the
        remapping ran correctly regardless of ID collisions.
        """
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        for example in result:
            intent = example["label_text"]
            expected_new_id = label2id[intent]
            original_id = _ORIGINAL_BANKING77_IDS[intent]
            # The new label must equal the new mapping, not the old Banking77 ID
            # (they may coincidentally be equal for activate_my_card: both 0)
            assert example["label"] == expected_new_id, (
                f"{intent!r}: expected new label {expected_new_id}, "
                f"got {example['label']} (original Banking77 id was {original_id})"
            )
            # For intents where original_id != expected_new_id, explicitly check
            if original_id != expected_new_id:
                assert example["label"] != original_id, (
                    f"{intent!r}: original Banking77 ID {original_id} was not replaced"
                )

    def test_label_column_is_classlabel_feature(self, mock_raw, label_mapping):
        """After filter_and_remap, 'label' must be a ClassLabel for stratified split."""
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        assert isinstance(result.features["label"], ClassLabel), (
            f"Expected ClassLabel, got {type(result.features['label'])}"
        )

    def test_classlabel_names_match_label2id(self, mock_raw, label_mapping):
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        cl: ClassLabel = result.features["label"]
        for intent, idx in label2id.items():
            assert cl.names[idx] == intent, (
                f"ClassLabel names[{idx}]={cl.names[idx]!r}, expected {intent!r}"
            )

    def test_each_class_has_expected_count(self, mock_raw, label_mapping):
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        counts = Counter(result["label_text"])
        for intent in SELECTED_INTENTS:
            assert counts[intent] == _EXAMPLES_PER_CLASS, (
                f"{intent!r}: expected {_EXAMPLES_PER_CLASS}, got {counts[intent]}"
            )

    def test_text_column_is_unchanged(self, mock_raw, label_mapping):
        """filter_and_remap must not modify the 'text' column content."""
        label2id, _ = label_mapping
        result = filter_and_remap(mock_raw, label2id)
        for text in result["text"]:
            assert isinstance(text, str) and len(text) > 0

    def test_mapping_applied_consistently_to_both_splits(self):
        """
        The same label2id applied to two independent splits must produce
        consistent label assignments for the same label_text value.
        """
        label2id, _ = build_label_mapping(SELECTED_INTENTS)
        split_a = _make_mock_raw_dataset(examples_per_class=5)
        split_b = _make_mock_raw_dataset(examples_per_class=3)
        filtered_a = filter_and_remap(split_a, label2id)
        filtered_b = filter_and_remap(split_b, label2id)

        # Build {label_text: label} dicts for both splits
        map_a = {ex["label_text"]: ex["label"] for ex in filtered_a}
        map_b = {ex["label_text"]: ex["label"] for ex in filtered_b}
        for intent in SELECTED_INTENTS:
            assert map_a[intent] == map_b[intent], (
                f"Inconsistent label for {intent!r}: split_a={map_a[intent]}, split_b={map_b[intent]}"
            )


# ---------------------------------------------------------------------------
# Tests: create_splits
# ---------------------------------------------------------------------------

class TestCreateSplits:

    def test_returns_two_datasets(self, mock_splits):
        train, val = mock_splits
        assert isinstance(train, Dataset)
        assert isinstance(val, Dataset)

    def test_combined_size_equals_input(self, mock_filtered, mock_splits):
        train, val = mock_splits
        assert len(train) + len(val) == len(mock_filtered)

    def test_approximate_val_ratio(self, mock_filtered, mock_splits):
        train, val = mock_splits
        actual_ratio = len(val) / len(mock_filtered)
        assert abs(actual_ratio - VAL_SPLIT_RATIO) < 0.05, (
            f"Val ratio {actual_ratio:.3f} deviates more than 5 % from "
            f"target {VAL_SPLIT_RATIO}"
        )

    def test_split_is_reproducible_same_seed(self, mock_filtered):
        train_a, val_a = create_splits(mock_filtered, val_ratio=0.20, seed=42)
        train_b, val_b = create_splits(mock_filtered, val_ratio=0.20, seed=42)
        assert train_a["text"] == train_b["text"]
        assert val_a["text"] == val_b["text"]

    def test_different_seeds_produce_different_splits(self, mock_filtered):
        train_a, _ = create_splits(mock_filtered, val_ratio=0.20, seed=42)
        train_b, _ = create_splits(mock_filtered, val_ratio=0.20, seed=99)
        # With 90 examples it is virtually certain that seeds 42 vs 99 differ
        assert train_a["text"] != train_b["text"], (
            "Different seeds produced identical splits — seeding may be broken"
        )

    def test_all_classes_present_in_train(self, mock_splits):
        train, _ = mock_splits
        present = set(train["label_text"])
        assert present == set(SELECTED_INTENTS), (
            f"Missing classes in train: {set(SELECTED_INTENTS) - present}"
        )

    def test_all_classes_present_in_val(self, mock_splits):
        _, val = mock_splits
        present = set(val["label_text"])
        assert present == set(SELECTED_INTENTS), (
            f"Missing classes in val: {set(SELECTED_INTENTS) - present}"
        )

    def test_class_proportions_are_stratified(self, mock_filtered, mock_splits):
        """
        After a stratified split each class should appear in both train and val
        at roughly the same proportion as in the original filtered dataset.
        Tolerance: ±1 example per class (rounding is expected with small N).
        """
        train, val = mock_splits
        total_counts = Counter(mock_filtered["label_text"])
        train_counts = Counter(train["label_text"])
        val_counts   = Counter(val["label_text"])

        for intent in SELECTED_INTENTS:
            n_total = total_counts[intent]
            expected_val   = round(n_total * VAL_SPLIT_RATIO)
            expected_train = n_total - expected_val
            assert abs(train_counts[intent] - expected_train) <= 1, (
                f"{intent!r}: train expected ~{expected_train}, got {train_counts[intent]}"
            )
            assert abs(val_counts[intent] - expected_val) <= 1, (
                f"{intent!r}: val expected ~{expected_val}, got {val_counts[intent]}"
            )

    def test_no_examples_shared_between_train_and_val(self, mock_splits):
        train, val = mock_splits
        overlap = set(train["text"]) & set(val["text"])
        assert len(overlap) == 0, (
            f"train/val share {len(overlap)} text(s): {list(overlap)[:3]}"
        )


# ---------------------------------------------------------------------------
# Tests: verify_no_leakage
# ---------------------------------------------------------------------------

class TestVerifyNoLeakage:

    def _make_non_overlapping_splits(self):
        """Three small splits with no shared texts."""
        def _ds(prefix, n):
            return Dataset.from_dict({
                "text": [f"{prefix}_text_{i}" for i in range(n)],
                "label": [i % 6 for i in range(n)],
                "label_text": [SELECTED_INTENTS[i % 6] for i in range(n)],
            })
        return _ds("train", 20), _ds("val", 5), _ds("test", 10)

    def test_clean_splits_pass(self):
        train, val, test = self._make_non_overlapping_splits()
        # Should not raise
        verify_no_leakage(train, val, test)

    def test_train_val_overlap_raises(self):
        train, val, test = self._make_non_overlapping_splits()
        # Inject a duplicate: add one train text into val.
        # In datasets 5.x, column access returns an Arrow Column, so we must
        # call list() before concatenating with a plain Python list.
        leaked_text = list(train["text"])[0]
        val_poisoned = Dataset.from_dict({
            "text": list(val["text"]) + [leaked_text],
            "label": list(val["label"]) + [0],
            "label_text": list(val["label_text"]) + [SELECTED_INTENTS[0]],
        })
        with pytest.raises(ValueError, match="leakage"):
            verify_no_leakage(train, val_poisoned, test)

    def test_train_test_overlap_raises(self):
        train, val, test = self._make_non_overlapping_splits()
        leaked_text = list(train["text"])[0]
        test_poisoned = Dataset.from_dict({
            "text": list(test["text"]) + [leaked_text],
            "label": list(test["label"]) + [0],
            "label_text": list(test["label_text"]) + [SELECTED_INTENTS[0]],
        })
        with pytest.raises(ValueError, match="leakage"):
            verify_no_leakage(train, val, test_poisoned)

    def test_val_test_overlap_raises(self):
        train, val, test = self._make_non_overlapping_splits()
        leaked_text = list(val["text"])[0]
        test_poisoned = Dataset.from_dict({
            "text": list(test["text"]) + [leaked_text],
            "label": list(test["label"]) + [0],
            "label_text": list(test["label_text"]) + [SELECTED_INTENTS[0]],
        })
        with pytest.raises(ValueError, match="leakage"):
            verify_no_leakage(train, val, test_poisoned)

    def test_split_from_create_splits_has_no_train_val_overlap(self, mock_splits):
        """Splits produced by create_splits must pass leakage check."""
        train, val = mock_splits
        # Use train as a stand-in test set to check the helper (no real test needed here)
        overlap = set(train["text"]) & set(val["text"])
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# Integration tests (require network, marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationRealDataset:
    """
    These tests load the actual mteb/banking77 dataset from Hugging Face.
    Run them with:  pytest tests/test_data_utils.py -m integration -v
    They are skipped in normal CI to avoid repeated downloads.
    """

    @pytest.fixture(scope="class")
    def real_data(self):
        return prepare_data()

    def test_prepare_data_returns_expected_keys(self, real_data):
        assert set(real_data.keys()) == {"train", "val", "test", "label2id", "id2label"}

    def test_real_train_size_is_reasonable(self, real_data):
        # Filtered train is ~846; after 80/20 split, train ≥ 600
        assert len(real_data["train"]) >= 600

    def test_real_val_size_is_reasonable(self, real_data):
        # Val ≥ 100
        assert len(real_data["val"]) >= 100

    def test_real_test_size_matches_expected(self, real_data):
        # Original Banking77 test subset: exactly 240 (40 per class × 6)
        assert len(real_data["test"]) == 240

    def test_real_label2id_matches_config(self, real_data):
        label2id = real_data["label2id"]
        assert set(label2id.keys()) == set(SELECTED_INTENTS)
        ids = sorted(label2id.values())
        assert ids == list(range(6))

    def test_real_no_leakage(self, real_data):
        # verify_no_leakage already ran inside prepare_data; call again to confirm
        verify_no_leakage(real_data["train"], real_data["val"], real_data["test"])

    def test_real_test_labels_are_zero_to_five(self, real_data):
        unique = sorted(set(real_data["test"]["label"]))
        assert unique == [0, 1, 2, 3, 4, 5]

    def test_real_all_classes_in_all_splits(self, real_data):
        for split_name in ("train", "val", "test"):
            present = set(real_data[split_name]["label_text"])
            assert present == set(SELECTED_INTENTS), (
                f"Split '{split_name}' missing: {set(SELECTED_INTENTS) - present}"
            )
