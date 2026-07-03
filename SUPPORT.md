# Support

## Before asking

1. Read the README — especially **Dataset**, **Installation** and
   **How to Run**; most setup problems are a missing `phishing_email.csv`
   in `dataset/raw/` or a missing dependency.
2. Check `outputs/` — the inspection reports and analysis files answer most
   "why is this number what it is" questions.
3. Search existing issues.

## Getting help

Open a **GitHub issue** with:

- your OS and Python version,
- the exact command you ran,
- the full traceback or the unexpected output,
- whether the dataset file is in `dataset/raw/`.

Questions about the methodology (leakage handling, the vectorizer tie
protocol, the tuning scope) are welcome as issues too — the design
decisions are documented in the module docstrings under `src/` and in the
IEEE paper.
