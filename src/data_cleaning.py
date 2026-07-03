"""
Dataset cleaning for Week 1.

Steps applied here, in order:
  1. Strip stray whitespace from the email text.
  2. Truncate extreme-length emails at MAX_EMAIL_CHARS. The longest raw
     email exceeds four million characters (a mailing-list digest); the
     cap keeps every email's informative opening while making the rest of
     the pipeline predictable. Truncation runs FIRST because two long
     emails that differ only beyond the cap become identical afterwards -
     deduplication must operate on the text the models will actually see.
  3. Drop rows that are exact duplicates of another row. Repeated emails
     could land in both the training and the testing split later and
     quietly inflate every score.
  4. Drop every copy of any text that appears under BOTH labels. An email
     stored once as phishing and once as legitimate is a labelling error;
     neither copy can be trusted, so both are removed.
  5. Drop remaining duplicate texts (same text, same label, stored twice).
  6. Drop rows with no usable text.
"""

from src import config


def clean_dataset(emails_df):
    """
    Purpose : Apply every Week 1 cleaning step to the dataset.
    Input   : raw DataFrame (text_combined, label).
    Output  : cleaned DataFrame with the same two columns.
    Return  : pandas.DataFrame
    """
    df = emails_df.copy()
    start_rows = len(df)

    # Step 1: whitespace cleanup
    df["text_combined"] = df["text_combined"].fillna("").str.strip()

    # Step 2: truncate extreme-length emails (before dedup, see docstring)
    too_long = df["text_combined"].str.len() > config.MAX_EMAIL_CHARS
    df.loc[too_long, "text_combined"] = (
        df.loc[too_long, "text_combined"].str.slice(0, config.MAX_EMAIL_CHARS)
    )
    print(f"Truncated over-length emails      : {too_long.sum():,} "
          f"(cap {config.MAX_EMAIL_CHARS:,} characters, "
          f"{100 * too_long.mean():.2f}% of the data)")

    # Step 3: exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed exact duplicate rows      : {before - len(df):,}")

    # Step 4: texts that appear under both labels
    label_counts = df.groupby("text_combined")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    before = len(df)
    df = df[~df["text_combined"].isin(conflicting_texts)]
    print(f"Removed label-conflicting rows    : {before - len(df):,} "
          f"({len(conflicting_texts):,} distinct texts)")

    # Step 5: remaining duplicate texts
    before = len(df)
    df = df.drop_duplicates(subset="text_combined")
    print(f"Removed duplicate-text rows       : {before - len(df):,}")

    # Step 6: rows with nothing left to learn from
    before = len(df)
    df = df[df["text_combined"] != ""]
    print(f"Removed empty-text rows           : {before - len(df):,}")

    df = df.reset_index(drop=True)
    print("Dataset Cleaning Completed")
    print(f"  Rows remaining: {len(df):,}")
    return df
