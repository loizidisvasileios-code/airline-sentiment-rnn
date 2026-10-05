"""Label encoding, dataset splitting and conversion of tokens into padded index sequences."""

from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import LABEL_COL, MAX_LEN_PERCENTILE, MIN_WORD_COUNT, PAD_IDX, RANDOM_SEED, UNK_IDX


def encode_labels(df):
    """Turn the three sentiment names into the integers CrossEntropyLoss expects."""
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df[LABEL_COL])
    class_names = list(encoder.classes_)
    print("Label mapping:", {name: index for index, name in enumerate(class_names)})
    return df, class_names


def split_data(df):
    """Split 80/10/10 into train, validation and test, preserving the class balance."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )
    print(f"Train {len(train_df)} | Validation {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def _build_vocabulary(token_lists):
    """Map every sufficiently frequent training word to an index, after the two reserved slots."""
    counts = Counter(token for tokens in token_lists for token in tokens)
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for word, count in counts.most_common():  # Ordered by frequency, so indices are stable
        if count >= MIN_WORD_COUNT:
            vocab[word] = len(vocab)
    return vocab


def _encode_sequences(token_lists, vocab, max_len):
    """Turn token lists into one padded integer matrix, and count what was lost."""
    matrix = np.full((len(token_lists), max_len), PAD_IDX, dtype=np.int64)  # Pre-filled with padding
    truncated = 0
    unknown = 0
    total = 0
    for row, tokens in enumerate(token_lists):
        truncated += len(tokens) > max_len
        for column, token in enumerate(tokens[:max_len]):  # Anything past max_len is dropped
            index = vocab.get(token, UNK_IDX)  # Words absent from the training vocabulary become <unk>
            matrix[row, column] = index
            unknown += index == UNK_IDX
            total += 1
    return matrix, truncated, unknown / total * 100


def build_sequence_features(train_tokens, val_tokens, test_tokens):
    """Build the vocabulary and the padding length from the training split only, then encode all three."""
    lengths = np.array([len(tokens) for tokens in train_tokens])
    max_len = int(np.percentile(lengths, MAX_LEN_PERCENTILE))
    print(
        f"Training lengths: mean {lengths.mean():.1f} | median {np.median(lengths):.0f} | "
        f"p{MAX_LEN_PERCENTILE} {max_len} | max {lengths.max()}"
    )

    vocab = _build_vocabulary(train_tokens)
    print(f"Vocabulary: {len(vocab)} entries (min_count={MIN_WORD_COUNT}, including <pad> and <unk>)")

    encoded = {}
    for name, tokens in [("train", train_tokens), ("validation", val_tokens), ("test", test_tokens)]:
        matrix, truncated, unknown_rate = _encode_sequences(tokens, vocab, max_len)
        encoded[name] = matrix
        print(
            f"{name:<11} {matrix.shape} | truncated {truncated} tweets "
            f"({truncated / len(tokens) * 100:.1f}%) | unknown tokens {unknown_rate:.1f}%"
        )

    return encoded["train"], encoded["validation"], encoded["test"], vocab, max_len
