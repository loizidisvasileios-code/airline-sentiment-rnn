"""Dataset loading, text cleaning and tokenization.

Carried over unchanged from Exercise 1, which is where every cleaning rule was measured
and justified. The only changes are the loader moved in here and the unused airline column
dropped, since this exercise forbids further exploratory analysis.
"""

import html
import re

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

from src.config import RAW_DATA_PATH, TEXT_COL, LABEL_COL

nltk.download("punkt_tab", quiet=True)  

# Patterns compiled once and reused for all 14,640 tweets.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")  # Links, present in 8.0% of tweets
MENTION_PATTERN = re.compile(r"@\w+")  # Twitter handles, present in 100% of tweets
CHAR_PATTERN = re.compile(r"[^a-z\s!?]")  # Anything that is not a letter, space, ! or ? which may have sentiment value. 
WHITESPACE_PATTERN = re.compile(r"\s+")  # One or more consecutive whitespace characters

# Contractions are expanded before punctuation is stripped, otherwise the apostrophe is
# removed and "can't" collapses to "can", inverting the meaning of the sentence.
CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not", "wasn't": "was not",
    "aren't": "are not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "ain't": "is not", "i'm": "i am", "it's": "it is",
    "that's": "that is", "what's": "what is", "he's": "he is", "she's": "she is",
    "there's": "there is", "here's": "here is", "you're": "you are", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "i'd": "i would", "let's": "let us",
}

# One pattern matching any key above; \b keeps it from firing inside a longer word
CONTRACTION_PATTERN = re.compile(r"\b(" + "|".join(CONTRACTIONS) + r")\b")


def load_raw_data():
    """Load the original dataset from disk without modifying it."""
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {RAW_DATA_PATH.name}")
    return df


def clean_text(text):
    """Clean a single tweet. The order of the steps matters and is explained below."""
    text = html.unescape(text)  #
    text = text.lower()  # Lowercase first so the patterns below only need to match lowercase
    text = text.replace("’", "'")  # Normalise the curly apostrophe phones insert
    text = CONTRACTION_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0)], text)  # can't -> can not
    text = URL_PATTERN.sub(" ", text)  # Remove links whole
    text = MENTION_PATTERN.sub(" ", text)  # Remove @handles whole
    text = text.replace("#", " ")  # Drop the hash symbol but keep the hashtag word itself
    text = CHAR_PATTERN.sub(" ", text)  # Removes digits, emojis and punctuation, keeping only ! and ?
    text = WHITESPACE_PATTERN.sub(" ", text).strip()  # Drop extra whitespace
    return text


def preprocess_dataset(df):
    """Clean, deduplicate and tokenize the dataset, reporting what was removed."""
    rows_in = len(df) 

    df = df[[TEXT_COL, LABEL_COL]].copy()  # Only the text and its label are used here
    df["clean_text"] = df[TEXT_COL].apply(clean_text)  

    empty = (df["clean_text"] == "").sum()  # Tweets that consisted only of noise
    df = df[df["clean_text"] != ""]  # Keep the rows that still have content

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)  # One row per distinct text
    duplicates = before_dedup - len(df)  # How many the deduplication removed

    df["tokens"] = df["clean_text"].apply(word_tokenize)  # Tokenize last, on the remaining rows only

    print(f"Rows in            : {rows_in}")
    print(f"Empty after cleaning: {empty}")
    print(f"Duplicates removed : {duplicates}")
    print(f"Rows out           : {len(df)}")
    return df
