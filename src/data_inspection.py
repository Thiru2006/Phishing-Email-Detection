"""
Dataset inspection.

Prints a summary of the dataset and saves the same summary as a text file
in outputs/. It runs twice during Week 1 (before and after cleaning) so
every cleaning decision is backed by measured numbers.

Beyond the usual shape/missing/duplicate checks, this inspection measures
content markers that decide how much text preprocessing is actually
needed: this corpus arrives partially normalised by its authors, and the
right engineering move is to verify that claim, not assume it. The marker
rates (URLs, HTML remnants, uppercase, digits) are the evidence. The
'enron' marker is tracked for the same reason the fake-news project
tracked '(Reuters)': a large share of the legitimate class comes from the
Enron corpus, and a source watermark is not the kind of signal a phishing
detector should be allowed to lean on.
"""

import pandas as pd

from src import config


def inspect_dataset(emails_df, stage, save_report=True):
    """
    Purpose : Show shape, columns, types, missing values, duplicates,
              length statistics, content markers and class balance.
    Input   : emails_df (DataFrame), stage (str, e.g. "raw" or "clean").
    Output  : Console summary + outputs/inspection_<stage>.txt
    Return  : None
    """
    lines = []

    def add(msg=""):
        print(msg)
        lines.append(str(msg))

    add(f"\n===== Dataset Inspection ({stage}) =====")
    add(f"Shape  : {emails_df.shape[0]:,} rows x {emails_df.shape[1]} columns")
    add(f"Columns: {list(emails_df.columns)}")

    add("\nData types:")
    add(emails_df.dtypes.to_string())

    add("\nMissing values per column:")
    add(emails_df.isnull().sum().to_string())

    add(f"\nExact duplicate rows: {emails_df.duplicated().sum():,}")

    text = emails_df["text_combined"].fillna("")
    add(f"Empty 'text_combined' entries: {(text.str.strip() == '').sum():,}")
    add(f"Duplicate 'text_combined' entries: {emails_df.duplicated(subset='text_combined').sum():,}")

    # Texts that appear under BOTH labels cannot be trusted as training data
    label_counts = emails_df.groupby("text_combined")["label"].nunique()
    conflicts = int((label_counts > 1).sum())
    add(f"Texts appearing under both labels: {conflicts:,}")

    lengths = text.str.len()
    add("\nEmail length in characters:")
    add(f"  mean {lengths.mean():,.0f} | median {lengths.median():,.0f} | "
        f"99th percentile {lengths.quantile(0.99):,.0f} | max {lengths.max():,}")

    add("\nContent markers (% of emails containing each):")
    markers = {
        "URL fragment ('http')": text.str.contains("http", regex=False),
        "HTML remnant ('<' and '>')": text.str.contains("<", regex=False)
        & text.str.contains(">", regex=False),
        "Uppercase letters": text.str.contains(r"[A-Z]", regex=True),
        "Digits": text.str.contains(r"\d", regex=True),
    }
    for name, mask in markers.items():
        add(f"  {name:<28}: {100 * mask.mean():5.1f}%")

    add("\n'enron' marker rate by class in % (1 = Phishing, 0 = Legitimate):")
    enron_rate = emails_df.groupby("label")["text_combined"].apply(
        lambda s: s.str.contains("enron", case=False, na=False).mean() * 100
    )
    add(enron_rate.round(1).to_string())

    add("\nClass distribution (1 = Phishing, 0 = Legitimate):")
    add(emails_df["label"].value_counts().to_string())

    if "clean_text" in emails_df.columns:
        clean = emails_df["clean_text"].fillna("")
        add("\n'clean_text' column checks:")
        add(f"  Empty entries: {(clean.str.strip() == '').sum():,}")
        add(f"  Average length: {clean.str.len().mean():,.0f} characters")

    if save_report:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = config.OUTPUTS_DIR / f"inspection_{stage}.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nInspection report saved -> {report_path}")
