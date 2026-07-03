"""
Loads the raw phishing email dataset.

The file ships as a single CSV with two columns: text_combined (the email
subject and body merged into one string by the dataset authors) and label.
The loader validates that shape instead of assuming it, because a wrong or
renamed column should stop the pipeline immediately rather than surface as
a confusing error three modules later.
"""

import sys

import pandas as pd

from src import config

EXPECTED_COLUMNS = {"text_combined", "label"}


def load_dataset():
    """
    Purpose : Load phishing_email.csv and validate its structure.
    Input   : None (path comes from config).
    Output  : DataFrame with columns text_combined, label.
    Return  : pandas.DataFrame
    """
    if not config.RAW_CSV.exists():
        sys.exit(
            f"ERROR: dataset file not found -> {config.RAW_CSV}\n"
            "Place phishing_email.csv inside dataset/raw/ and run again."
        )

    emails_df = pd.read_csv(config.RAW_CSV)

    missing = EXPECTED_COLUMNS - set(emails_df.columns)
    if missing:
        sys.exit(f"ERROR: dataset is missing expected column(s): {missing}")

    emails_df["label"] = emails_df["label"].astype(int)

    phishing = (emails_df["label"] == config.LABEL_PHISHING).sum()
    legit = (emails_df["label"] == config.LABEL_LEGIT).sum()

    print("Dataset Loaded Successfully")
    print(f"  Phishing emails (1) : {phishing:,}")
    print(f"  Legitimate emails (0): {legit:,}")
    print(f"  Total emails         : {len(emails_df):,}")
    return emails_df
