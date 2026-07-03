"""
Central configuration for the Phishing Email Detection project.

Every path and constant lives here so no other module hard-codes anything.
All paths are built relative to the project root, so the project runs the
same way on any machine after phishing_email.csv is placed in dataset/raw/.
"""

from pathlib import Path

# Project root = the folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset locations
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

RAW_CSV = RAW_DIR / "phishing_email.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_emails.csv"

# Output locations
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"

# Label convention used everywhere in this project:
# 1 = Phishing email (the class we want to detect), 0 = Legitimate email
LABEL_PHISHING = 1
LABEL_LEGIT = 0

# Emails longer than this many characters are truncated during cleaning.
# The corpus contains a handful of extreme outliers (the longest raw email
# is over 4 million characters - a mailing-list digest, not a message a
# classifier should ever need in full). The cap keeps the informative
# opening of every email while making preprocessing and vectorization
# predictable; the cleaning step prints exactly how many emails it touched.
MAX_EMAIL_CHARS = 20_000

# Fixed seed so every run gives the same result
RANDOM_STATE = 42

# Week 2: feature engineering settings
MAX_FEATURES = 5000   # vocabulary size for both vectorizers (matches the
                      # convention used across both internship projects)
TEST_SIZE = 0.2       # 80/20 train-test split

# Week 2: generated evidence files
EDA_SUMMARY_TXT = OUTPUTS_DIR / "eda_summary.txt"
FEATURE_NOTES_TXT = OUTPUTS_DIR / "feature_engineering_notes.txt"
VECTORIZER_COMPARISON_CSV = OUTPUTS_DIR / "vectorizer_comparison.csv"

# Week 3: model building
MODEL_COMPARISON_CSV = OUTPUTS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORTS_TXT = OUTPUTS_DIR / "classification_reports.txt"
BEST_MODEL_TXT = OUTPUTS_DIR / "best_model_analysis.txt"
TUNING_CSV = OUTPUTS_DIR / "hyperparameter_tuning.csv"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
