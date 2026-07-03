"""
Feature engineering for Week 2.

Order matters in this module. The dataset is split into training and
testing sets FIRST, and every vectorizer is fitted on the training emails
only. Fitting a vectorizer on the full corpus would leak vocabulary and
document-frequency statistics from the test emails into training, which
quietly inflates the final scores.

The CountVectorizer vs TF-IDF comparison is done with a quick Logistic
Regression probe using 3-fold cross-validation on the training set only.
The test set stays untouched until the Week 3 model evaluation. Whichever
representation measures better goes forward; the decision text is written
from the numbers, not from habit.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src import config

# Two mean-F1 scores closer than this are treated as a tie
TIE_THRESHOLD = 0.001


def split_data(df):
    """
    Purpose : Stratified 80/20 split of the cleaned dataset.
    Input   : DataFrame with clean_text and label columns.
    Output  : X_train_text, X_test_text, y_train, y_test (+ printed note).
    Return  : tuple
    """
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=config.TEST_SIZE,
        stratify=df["label"],
        random_state=config.RANDOM_STATE,
    )

    train_phish = 100 * (y_train == config.LABEL_PHISHING).mean()
    test_phish = 100 * (y_test == config.LABEL_PHISHING).mean()

    print("Train/Test Split Completed (stratified)")
    explanation = (
        f"Split note: an 80/20 split keeps {len(y_train):,} emails for training and "
        f"holds out {len(y_test):,} for testing. 20% is enough here because over "
        f"16,000 unseen emails give very stable metric estimates, while 80% leaves "
        f"the models plenty to learn from. The split is stratified on the label, so "
        f"the phishing share is {train_phish:.1f}% in training and {test_phish:.1f}% "
        f"in testing - neither side is accidentally easier than the other. "
        f"random_state={config.RANDOM_STATE} makes the split reproducible on every run."
    )
    print(explanation)
    return X_train_text, X_test_text, y_train, y_test, explanation


def compare_vectorizers(X_train_text, y_train):
    """
    Purpose : Compare CountVectorizer and TF-IDF with a Logistic Regression
              probe (3-fold CV on the training set only) and pick one.
    Input   : training texts and labels.
    Output  : outputs/vectorizer_comparison.csv, graphs/vectorizer_comparison.png,
              printed justification.
    Return  : (chosen_name, justification_text)
    """
    print("\nComparing CountVectorizer vs TF-IDF (3-fold CV on training set) ...")

    vectorizers = {
        "CountVectorizer": CountVectorizer(max_features=config.MAX_FEATURES),
        "TF-IDF": TfidfVectorizer(max_features=config.MAX_FEATURES),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    for name, vectorizer in vectorizers.items():
        X = vectorizer.fit_transform(X_train_text)
        probe = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
        scores = cross_validate(probe, X, y_train, cv=cv, scoring=["accuracy", "f1"])
        rows.append(
            {
                "vectorizer": name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_f1_mean": scores["test_f1"].mean(),
                "cv_f1_std": scores["test_f1"].std(),
                "mean_fit_time_s": scores["fit_time"].mean(),
            }
        )
        print(
            f"  {name:<16} CV accuracy {rows[-1]['cv_accuracy_mean']:.4f} "
            f"(+/- {rows[-1]['cv_accuracy_std']:.4f}) | CV F1 "
            f"{rows[-1]['cv_f1_mean']:.4f} (+/- {rows[-1]['cv_f1_std']:.4f})"
        )

    comparison = pd.DataFrame(rows)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.VECTORIZER_COMPARISON_CSV, index=False)
    print(f"Comparison table saved -> {config.VECTORIZER_COMPARISON_CSV}")

    # Small chart of the comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(comparison))
    width = 0.35
    ax.bar([i - width / 2 for i in x], comparison["cv_accuracy_mean"], width,
           label="CV Accuracy", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], comparison["cv_f1_mean"], width,
           label="CV F1 (phishing class)", color="#ff7f0e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["vectorizer"])
    low = min(comparison["cv_accuracy_mean"].min(), comparison["cv_f1_mean"].min())
    ax.set_ylim(low - 0.01, 1.0)
    ax.set_title("Vectorizer Comparison (Logistic Regression probe, 3-fold CV)")
    ax.set_ylabel("Score")
    ax.legend()
    for i, row in comparison.iterrows():
        ax.text(i - width / 2, row["cv_accuracy_mean"], f"{row['cv_accuracy_mean']:.4f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["cv_f1_mean"], f"{row['cv_f1_mean']:.4f}",
                ha="center", va="bottom", fontsize=8)
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.GRAPHS_DIR / "vectorizer_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved -> {config.GRAPHS_DIR / 'vectorizer_comparison.png'}")

    # Decide, and explain the decision from the measured numbers
    best = comparison.sort_values("cv_f1_mean", ascending=False).iloc[0]
    other = comparison.sort_values("cv_f1_mean", ascending=False).iloc[1]
    diff = best["cv_f1_mean"] - other["cv_f1_mean"]

    if diff <= TIE_THRESHOLD:
        chosen = "TF-IDF"
        justification = (
            f"Vectorizer decision: the probe scores are effectively tied "
            f"(F1 {best['cv_f1_mean']:.4f} vs {other['cv_f1_mean']:.4f}, a gap of "
            f"{diff:.4f}). Since the measured difference is within noise, TF-IDF is "
            f"chosen on principle: it down-weights words that appear in almost every "
            f"email and emphasises distinctive vocabulary, which usually generalises "
            f"better than raw counts, and it is the representation suggested in the "
            f"project PDF. All Week 3 models will therefore train on TF-IDF features."
        )
    else:
        chosen = best["vectorizer"]
        justification = (
            f"Vectorizer decision: {chosen} scored higher in the probe "
            f"(F1 {best['cv_f1_mean']:.4f} vs {other['cv_f1_mean']:.4f}, accuracy "
            f"{best['cv_accuracy_mean']:.4f} vs {other['cv_accuracy_mean']:.4f}) with "
            f"3-fold cross-validation on the training set, so it is selected for "
            f"Week 3. The comparison used the same vocabulary size "
            f"({config.MAX_FEATURES:,} features) and the same Logistic Regression "
            f"probe for both representations, so the difference comes from the "
            f"features themselves and not from the model setup."
        )

    print("\n" + justification)
    return chosen, justification


def build_features(X_train_text, X_test_text, chosen):
    """
    Purpose : Fit the chosen vectorizer on training text only and transform
              both splits into feature matrices.
    Input   : training texts, testing texts, chosen vectorizer name.
    Output  : X_train, X_test sparse matrices + the fitted vectorizer.
    Return  : (X_train, X_test, vectorizer)
    """
    if chosen == "TF-IDF":
        vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES)
    else:
        vectorizer = CountVectorizer(max_features=config.MAX_FEATURES)

    X_train = vectorizer.fit_transform(X_train_text)  # fit on training only
    X_test = vectorizer.transform(X_test_text)        # transform, never fit

    print("\nFeature Extraction Completed")
    print(f"  Vectorizer      : {chosen}")
    print(f"  Training matrix : {X_train.shape[0]:,} emails x {X_train.shape[1]:,} features")
    print(f"  Testing matrix  : {X_test.shape[0]:,} emails x {X_test.shape[1]:,} features")
    return X_train, X_test, vectorizer
