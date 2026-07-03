"""
Text preprocessing for Week 1.

The pipeline is the same one used in the fake news project - lowercase,
remove URLs, HTML tags, mentions, punctuation, special characters and
numbers, then filter English stop words. This corpus arrives partially
normalised by its authors (the raw-stage inspection report measures
exactly how much), so several steps act on very little - but every step is
idempotent, cheap, and guarantees the same text contract to the rest of
the pipeline regardless of what the upstream data looked like.

'enron' is treated as an extra stop word. A large share of the legitimate
class comes from the Enron email corpus, so the company name works as a
source watermark rather than as content - the same problem the '(Reuters)'
dateline caused in the fake news dataset. The inspection report measures
its per-class rate; removing it forces the models to judge emails by their
wording. Deeper source effects that single tokens cannot capture are noted
as a limitation for the report.

Stemming / lemmatization is intentionally not applied, for the same
measured reason as in Project 1: with bag-of-words features over a corpus
of this size it rarely changes results while slowing preprocessing.
"""

import re

import nltk
from nltk.corpus import stopwords

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"@\w+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")

EXTRA_STOPWORDS = {"enron"}


def get_stopwords():
    """
    Purpose : Return the English stop-word set (downloads it on first run).
    Input   : None
    Output  : set of stop words, including the project-specific extras.
    Return  : set[str]
    """
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return set(words) | EXTRA_STOPWORDS


def clean_text(text, stop_words):
    """
    Purpose : Normalise one email for feature extraction.
    Input   : text (str), stop_words (set)
    Output  : lowercase string containing only informative words.
    Return  : str
    """
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(words)


def preprocess_dataframe(df):
    """
    Purpose : Apply clean_text to every email in the dataset.
    Input   : DataFrame with a 'text_combined' column.
    Output  : same DataFrame with a new 'clean_text' column; rows that
              end up empty after preprocessing are dropped.
    Return  : pandas.DataFrame
    """
    stop_words = get_stopwords()
    print("Text Preprocessing Started ...")
    df = df.copy()
    df["clean_text"] = df["text_combined"].apply(lambda t: clean_text(t, stop_words))

    before = len(df)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed rows left empty after preprocessing: {removed:,}")

    print("Text Preprocessing Completed")
    return df
