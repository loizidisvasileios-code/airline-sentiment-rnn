"""Central configuration: file paths, random seed, column names and sequence settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Tweets.csv"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Trained weights, kept out of version control
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Single random seed for reproducibility
RANDOM_SEED = 42

TEXT_COL = "text"

LABEL_COL = "airline_sentiment"

# Reserved vocabulary slots. PAD_IDX must stay 0: nn.Embedding(padding_idx=0) relies on it,
# and the attention mask is built by testing the input against it.
PAD_IDX = 0
UNK_IDX = 1

# Sequences are padded to this percentile of the training lengths rather than to the
# maximum, so the few longest tweets do not force padding onto every other one.
MAX_LEN_PERCENTILE = 95

# Words seen fewer times than this in the training split stay out of the vocabulary
MIN_WORD_COUNT = 2
