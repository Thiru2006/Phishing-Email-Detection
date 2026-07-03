"""
Phishing Email Detection - main pipeline.

Week 1: load -> inspect raw -> clean -> preprocess -> inspect clean -> save
Week 2: EDA -> stratified split -> vectorizer comparison -> features

Week 3 (model building) and Week 4 (deliverables) will be added here so
the whole project always reproduces with a single command:

    python main.py
"""

from src import config
from src.data_loader import load_dataset
from src.data_inspection import inspect_dataset
from src.data_cleaning import clean_dataset
from src.text_preprocessing import preprocess_dataframe
from src.eda import run_eda
from src.feature_engineering import build_features, compare_vectorizers, split_data
from src.model_training import get_models, slug, train_model, tune_models
from src.model_evaluation import (
    compare_models,
    evaluate_model,
    identify_best_model,
    plot_feature_importance,
    save_reports,
)

import joblib
import pandas as pd


def main():
    print("========== PHISHING EMAIL DETECTION : WEEK 1 ==========")

    # 1. Load and validate the raw file
    emails_df = load_dataset()

    # 2. Inspect the raw data (report saved to outputs/)
    inspect_dataset(emails_df, stage="raw")

    # 3. Clean the dataset
    emails_df = clean_dataset(emails_df)

    # 4. Preprocess the email text
    emails_df = preprocess_dataframe(emails_df)

    # 5. Inspect the cleaned data to confirm every problem was handled
    inspect_dataset(emails_df, stage="clean")

    # 6. Save the cleaned dataset (the original raw file stays untouched)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    emails_df.to_csv(config.CLEAN_CSV, index=False)
    print(f"\nCleaned dataset saved -> {config.CLEAN_CSV}")
    print(f"Final shape: {emails_df.shape[0]:,} rows x {emails_df.shape[1]} columns")
    print("========== WEEK 1 COMPLETED ==========")

    # ------------------------- WEEK 2 -------------------------
    print("\n========== PHISHING EMAIL DETECTION : WEEK 2 ==========")

    run_eda(emails_df)

    X_train_text, X_test_text, y_train, y_test, split_note = split_data(emails_df)
    chosen, vec_note = compare_vectorizers(X_train_text, y_train)
    X_train, X_test, vectorizer = build_features(X_train_text, X_test_text, chosen)

    # Keep the written feature-engineering decisions next to the graphs
    config.FEATURE_NOTES_TXT.write_text(
        split_note + "\n\n" + vec_note, encoding="utf-8"
    )
    print(f"Feature engineering notes saved -> {config.FEATURE_NOTES_TXT}")

    print("========== WEEK 2 COMPLETED ==========")

    # ------------------------- WEEK 3 -------------------------
    print("\n========== PHISHING EMAIL DETECTION : WEEK 3 ==========")

    tuned = tune_models(X_train, y_train)
    models = get_models(lr_C=tuned["lr_C"], nb_alpha=tuned["nb_alpha"])

    fitted = {}
    rows = []
    report_blocks = []
    for name, model in models.items():
        model, train_time = train_model(name, model, X_train, y_train)
        fitted[name] = model
        row, report_block = evaluate_model(name, model, X_test, y_test)
        row["train_time_s"] = train_time
        rows.append(row)
        report_blocks.append(report_block)

    results_df = pd.DataFrame(rows)
    save_reports(report_blocks)
    compare_models(results_df)
    best_name = identify_best_model(results_df, tuned)

    # Feature importance figure (project brief requirement) from the
    # interpretable linear model's coefficients
    plot_feature_importance(fitted["Logistic Regression"], vectorizer)

    # Save the winning model and the fitted vectorizer for reuse
    joblib.dump(fitted[best_name], config.BEST_MODEL_FILE, compress=3)
    joblib.dump(vectorizer, config.VECTORIZER_FILE, compress=3)
    print(f"Best model saved -> {config.BEST_MODEL_FILE}")
    print(f"Vectorizer saved -> {config.VECTORIZER_FILE}")

    print("========== WEEK 3 COMPLETED ==========")


if __name__ == "__main__":
    main()
