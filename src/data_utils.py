"""
data_utils.py — Dataset loading, verification, filtering, splitting, and
leakage detection for the Banking77 6-class intent classifier.

Public API
----------
    load_and_verify_dataset() -> DatasetDict
    build_label_mapping(selected_intents) -> tuple[dict, dict]
    filter_and_remap(split, label2id) -> Dataset
    create_splits(filtered_train, val_ratio, seed) -> tuple[Dataset, Dataset]
    verify_no_leakage(train, val, test) -> None   # raises ValueError on leakage
    prepare_data() -> dict                        # orchestrates all steps

Run as a script
---------------
    python -m src.data_utils

    Executes the full pipeline, prints a verification report, and exits.
    Does NOT train anything or download the pretrained model.
"""

from __future__ import annotations

from collections import Counter

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from src.config import (
    DATASET_ID,
    SEED,
    SELECTED_INTENTS,
    VAL_SPLIT_RATIO,
)

# ---------------------------------------------------------------------------
# Step 1: Load and verify
# ---------------------------------------------------------------------------

def load_and_verify_dataset() -> DatasetDict:
    """
    Load mteb/banking77 and verify:
      - Schema contains 'text', 'label', 'label_text' columns.
      - Both 'train' and 'test' splits are present with expected sizes.
      - All six SELECTED_INTENTS exist in the dataset (matched by string).

    Prints a detailed report to stdout. Raises ValueError on any mismatch.
    """
    print(f"Loading dataset '{DATASET_ID}'...")
    dataset: DatasetDict = load_dataset(DATASET_ID)

    # --- Schema check ---
    required_columns = {"text", "label", "label_text"}
    for split_name in ("train", "test"):
        if split_name not in dataset:
            raise ValueError(
                f"Expected split '{split_name}' not found. "
                f"Available: {list(dataset.keys())}"
            )
        actual = set(dataset[split_name].column_names)
        missing = required_columns - actual
        if missing:
            raise ValueError(
                f"Schema mismatch in '{split_name}' split. "
                f"Missing columns: {missing}. Got: {actual}"
            )

    train_split = dataset["train"]
    test_split  = dataset["test"]

    print(f"\n  Dataset: {DATASET_ID}")
    print(f"  Schema:  {dict(train_split.features)}")
    print(f"  Splits:")
    print(f"    train : {len(train_split):>6,} examples")
    print(f"    test  : {len(test_split):>6,} examples")

    # --- Label name discovery ---
    # mteb/banking77 stores labels as raw int64; human-readable names are
    # in the 'label_text' column. We cannot use .features["label"].names.
    all_intents_in_train = sorted(set(train_split["label_text"]))
    all_intents_in_test  = sorted(set(test_split["label_text"]))

    print(f"\n  Unique intents in train : {len(all_intents_in_train)}")
    print(f"  Unique intents in test  : {len(all_intents_in_test)}")

    # --- Verify selected six intents exist (by string, never by ID) ---
    print("\n  Verifying selected intents against actual dataset labels...")
    train_counts = Counter(train_split["label_text"])
    test_counts  = Counter(test_split["label_text"])

    missing_from_train = [i for i in SELECTED_INTENTS if i not in all_intents_in_train]
    missing_from_test  = [i for i in SELECTED_INTENTS if i not in all_intents_in_test]
    if missing_from_train:
        raise ValueError(
            f"Selected intents missing from train split: {missing_from_train}\n"
            f"Available: {all_intents_in_train}"
        )
    if missing_from_test:
        raise ValueError(
            f"Selected intents missing from test split: {missing_from_test}\n"
            f"Available: {all_intents_in_test}"
        )

    print(f"  {'Intent':<48} {'Train':>6} {'Test':>6}")
    print(f"  {'-'*48} {'-'*6} {'-'*6}")
    for intent in SELECTED_INTENTS:
        print(f"  {intent:<48} {train_counts[intent]:>6} {test_counts[intent]:>6}")
    print("  All six intents verified ✓")

    return dataset


# ---------------------------------------------------------------------------
# Step 2: Build label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(
    selected_intents: list[str] = SELECTED_INTENTS,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build label2id and id2label from the given list of intent strings.

    The input list must already be in the desired label order (alphabetical
    in our case, as defined in config.SELECTED_INTENTS). We do NOT sort here
    so callers retain full control over ordering.

    Example:
        label2id["activate_my_card"]  → 0
        id2label[0]                   → "activate_my_card"
    """
    label2id: dict[str, int] = {intent: idx for idx, intent in enumerate(selected_intents)}
    id2label: dict[int, str] = {idx: intent for intent, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Step 3: Filter and remap
# ---------------------------------------------------------------------------

def filter_and_remap(split: Dataset, label2id: dict[str, int]) -> Dataset:
    """
    Filter a dataset split to only the intents in label2id, then replace
    the 'label' column with new integer IDs (0 … N-1) from label2id.

    The result is cast so that 'label' becomes a ClassLabel feature, which
    enables stratified splitting via train_test_split(stratify_by_column=...).

    Parameters
    ----------
    split    : A single Dataset split (train or test).
    label2id : Mapping of intent-string → new integer label (0-based).

    Returns
    -------
    A Dataset containing only the selected intents with remapped labels.
    """
    selected = set(label2id.keys())

    # Filter rows whose label_text is one of the six selected intents.
    filtered = split.filter(
        lambda example: example["label_text"] in selected,
        desc="Filtering intents",
    )

    # Replace the original Banking77 integer label with our 0-based ID.
    def _remap(example: dict) -> dict:
        example["label"] = label2id[example["label_text"]]
        return example

    filtered = filtered.map(_remap, desc="Remapping labels")

    # Cast 'label' to ClassLabel so datasets can stratify on it.
    # names list: index i → intent string (same order as label2id).
    names = [intent for intent, _ in sorted(label2id.items(), key=lambda x: x[1])]
    new_features = filtered.features.copy()
    new_features["label"] = ClassLabel(num_classes=len(names), names=names)
    filtered = filtered.cast(new_features)

    return filtered


# ---------------------------------------------------------------------------
# Step 4: Stratified train/validation split
# ---------------------------------------------------------------------------

def create_splits(
    filtered_train: Dataset,
    val_ratio: float = VAL_SPLIT_RATIO,
    seed: int = SEED,
) -> tuple[Dataset, Dataset]:
    """
    Create a reproducible, stratified train / validation split from the
    filtered training data.

    The original test split is NOT touched here — it remains reserved for
    final evaluation only.

    Parameters
    ----------
    filtered_train : Filtered and remapped training Dataset (label is ClassLabel).
    val_ratio      : Fraction of examples to place in the validation set.
    seed           : Random seed for reproducibility.

    Returns
    -------
    (train_split, val_split) — two Dataset objects.
    """
    split_result = filtered_train.train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )
    # train_test_split returns {"train": ..., "test": ...}.
    # We rename "test" → validation to avoid confusion with the held-out test set.
    return split_result["train"], split_result["test"]


# ---------------------------------------------------------------------------
# Step 5: Leakage detection
# ---------------------------------------------------------------------------

def verify_no_leakage(train: Dataset, val: Dataset, test: Dataset) -> None:
    """
    Check for text overlap between all three splits.

    Reports the size of every overlap and — if any exists — raises
    ValueError with a summary. Overlap text examples are printed so you
    can inspect them before deciding how to handle them.

    Parameters
    ----------
    train, val, test : The three Dataset splits.

    Raises
    ------
    ValueError if any cross-split text overlap is found.
    """
    print("\n  Checking for text overlap across splits...")

    train_texts = set(train["text"])
    val_texts   = set(val["text"])
    test_texts  = set(test["text"])

    pairs = {
        "train ∩ val":  (train_texts, val_texts),
        "train ∩ test": (train_texts, test_texts),
        "val   ∩ test": (val_texts,   test_texts),
    }

    any_overlap = False
    for pair_name, (a, b) in pairs.items():
        overlap = a & b
        if overlap:
            any_overlap = True
            print(f"  ⚠ OVERLAP in {pair_name}: {len(overlap)} text(s)")
            for text in list(overlap)[:3]:
                print(f"      {text!r}")
            if len(overlap) > 3:
                print(f"      … and {len(overlap) - 3} more")

    if any_overlap:
        raise ValueError(
            "Data leakage detected: one or more splits share identical texts. "
            "See the details printed above."
        )

    print(f"  No overlap detected ✓")
    print(f"    train : {len(train_texts):>5} unique texts")
    print(f"    val   : {len(val_texts):>5} unique texts")
    print(f"    test  : {len(test_texts):>5} unique texts")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare_data() -> dict:
    """
    Run the full data preparation pipeline:

        load → verify → build mapping → filter & remap → split → leakage check

    Returns a dict with keys:
        "train"    : Dataset  (training split, ~80 % of filtered train)
        "val"      : Dataset  (validation split, ~20 % of filtered train)
        "test"     : Dataset  (original test split, held out for final eval)
        "label2id" : dict[str, int]
        "id2label" : dict[int, str]
    """
    print("=" * 60)
    print("  DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1
    dataset = load_and_verify_dataset()

    # Step 2
    label2id, id2label = build_label_mapping(SELECTED_INTENTS)
    print(f"\n  Label mapping (alphabetical, 0-based):")
    for intent, idx in label2id.items():
        print(f"    {idx}: {intent}")

    # Step 3 — filter and remap both splits independently using the SAME mapping
    print("\n  Filtering and remapping splits...")
    filtered_train = filter_and_remap(dataset["train"], label2id)
    filtered_test  = filter_and_remap(dataset["test"],  label2id)
    print(f"    Filtered train : {len(filtered_train):>5} examples")
    print(f"    Filtered test  : {len(filtered_test):>5} examples")

    # Step 4 — stratified train/val split from filtered_train only
    print(f"\n  Creating stratified {int((1-VAL_SPLIT_RATIO)*100)}/{int(VAL_SPLIT_RATIO*100)} "
          f"train/val split (seed={SEED})...")
    train_split, val_split = create_splits(filtered_train, VAL_SPLIT_RATIO, SEED)

    # Step 5 — leakage detection
    verify_no_leakage(train_split, val_split, filtered_test)

    # Summary
    print("\n" + "=" * 60)
    print("  SPLIT SUMMARY")
    print("=" * 60)
    print(f"  {'Split':<8} {'Size':>6}  {'Per-class counts'}")
    print(f"  {'-'*8} {'-'*6}  {'-'*40}")
    for name, ds in [("train", train_split), ("val", val_split), ("test", filtered_test)]:
        counts = Counter(ds["label"])
        count_str = "  ".join(
            f"{id2label[i]}={counts[i]}" for i in sorted(counts)
        )
        print(f"  {name:<8} {len(ds):>6}  {count_str}")

    print("\n  Data preparation complete ✓")

    return {
        "train":    train_split,
        "val":      val_split,
        "test":     filtered_test,
        "label2id": label2id,
        "id2label": id2label,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = prepare_data()
    print(f"\nReturned keys: {list(result.keys())}")
