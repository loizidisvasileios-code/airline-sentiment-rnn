"""Dataset loading, text cleaning and tokenization.
"""

import html
import re

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

from src.config import RAW_DATA_PATH, TEXT_COL, LABEL_COL

nltk.download("punkt_tab", quiet=True)  


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")  
MENTION_PATTERN = re.compile(r"@\w+")  
CHAR_PATTERN = re.compile(r"[^a-z\s!?]")
WHITESPACE_PATTERN = re.compile(r"\s+")  


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


CONTRACTION_PATTERN = re.compile(r"\b(" + "|".join(CONTRACTIONS) + r")\b")


def load_raw_data():
    """Load the original dataset from disk without modifying it."""
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {RAW_DATA_PATH.name}")
    return df


def clean_text(text):
    """Clean a single tweet. The order of the steps matters and is explained below."""
    text = html.unescape(text)  
    text = text.lower()  
    text = text.replace("’", "'") 
    text = CONTRACTION_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0)], text)  
    text = URL_PATTERN.sub(" ", text)  
    text = MENTION_PATTERN.sub(" ", text)  
    text = text.replace("#", " ")  
    text = CHAR_PATTERN.sub(" ", text)  
    text = WHITESPACE_PATTERN.sub(" ", text).strip()  
    return text


def preprocess_dataset(df):
    """Clean, deduplicate and tokenize the dataset."""
    rows_in = len(df) 

    df = df[[TEXT_COL, LABEL_COL]].copy()  
    df["clean_text"] = df[TEXT_COL].apply(clean_text)  

    empty = (df["clean_text"] == "").sum()  
    df = df[df["clean_text"] != ""]  

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)  
    duplicates = before_dedup - len(df)  

    df["tokens"] = df["clean_text"].apply(word_tokenize)  

    print(f"Rows in            : {rows_in}")
    print(f"Empty after cleaning: {empty}")
    print(f"Duplicates removed : {duplicates}")
    print(f"Rows out           : {len(df)}")
    return df
