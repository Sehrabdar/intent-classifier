"""
config.py — Single source of truth for all project constants.

No logic lives here. Import from this module in all other src files.
"""

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# PolyAI/banking77 cannot be loaded with datasets>=2.x (script-based, broken).
# mteb/banking77 is the verified working source: same data, Parquet format.
DATASET_ID = "mteb/banking77"

# Six intents selected for diversity across card management, security,
# account balance, money movement, and currency domains.
# Kept in alphabetical order so enumerate() produces label IDs 0–5 consistently.
SELECTED_INTENTS: list[str] = [
    "activate_my_card",                      # label 0
    "balance_not_updated_after_bank_transfer",  # label 1
    "card_arrival",                          # label 2
    "exchange_rate",                         # label 3
    "lost_or_stolen_card",                   # label 4
    "request_refund",                        # label 5
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert/distilbert-base-uncased"

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

SEED = 42
MAX_LENGTH = 64        # Banking77 queries are short; 64 tokens is ample
VAL_SPLIT_RATIO = 0.20  # 20 % of filtered train → validation

# ---------------------------------------------------------------------------
# Training (not used until Milestone 2)
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FP16 = True  # Will be smoke-tested before full training

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/intent-classifier"
