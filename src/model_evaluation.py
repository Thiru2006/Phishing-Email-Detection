"""
Model evaluation for Week 3.

Every model is evaluated on the same held-out test set with the metrics
required by the project: accuracy, precision, recall, F1 score, confusion
matrix and the full classification report. Precision/recall/F1 are
reported for the phishing class (label 1), because catching malicious
mail is the point of the system - and recall gets special attention in
the analysis, since a missed phishing email (false negative) costs a user
far more than a false alarm costs a reviewer.

The module also builds the comparison charts, decides which model
performed best from the measured numbers, writes a plain-language
analysis of why, and produces the feature-importance figure the project
brief asks for - the most indicative words on each side of the decision,
read from the logistic regression coefficients.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.model_training import slug

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]

# Why each algorithm behaves the way it does on this kind of data.
# The final analysis combines one of these with the measured numbers.
MODEL_REASONS = {
    "Logistic Regression": (
        "it learns one weight per word and adds the evidence up in log-odds. "
        "The two email vocabularies barely overlap (the Week 2 top-word "
        "analysis found only 7 shared words in the top 20), so a linear "
        "boundary over 5,000 TF-IDF features separates them cleanly, and L2 "
        "regularisation keeps the weights small enough to generalise"
    ),
    "Multinomial Naive Bayes": (
        "its generative model is built for word-frequency data: it estimates "
        "how often each word appears in phishing versus legitimate mail and "
        "multiplies the evidence. The independence assumption is wrong for "
        "language, but with 65k training emails the per-word statistics are "
        "strong, and smoothing was tuned rather than defaulted"
    ),
    "Random Forest": (
        "it averages a hundred decorrelated trees, which removes single-tree "
        "overfitting and captures word interactions a linear model cannot. "
        "Its cost on text is that each tree only sees a random slice of the "
        "5,000 words, so many trees are needed and training is the slowest "
        "of the four"
    ),
    "MLP Neural Network": (
        "its hidden layer learns non-linear combinations of TF-IDF features, "
        "giving it more capacity than any linear model here. When the classes "
        "are already close to linearly separable, that extra capacity mostly "
        "re-draws the same boundary, so it competes with - rather than "
        "dominates - logistic regression, at a higher training cost"
    ),
}


def evaluate_model(name, model, X_test, y_test):
    """
    Purpose : Evaluate one fitted model on the shared test set.
    Input   : display name, fitted model, test matrix + labels.
    Output  : confusion matrix PNG in graphs/, printed metrics.
    Return  : (metrics row dict, classification report text)
    """
    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=config.LABEL_PHISHING),
        "recall": recall_score(y_test, predictions, pos_label=config.LABEL_PHISHING),
        "f1": f1_score(y_test, predictions, pos_label=config.LABEL_PHISHING),
        "predict_time_s": predict_time,
    }

    cm = confusion_matrix(
        y_test, predictions, labels=[config.LABEL_LEGIT, config.LABEL_PHISHING]
    )
    (row["tn"], row["fp"]), (row["fn"], row["tp"]) = cm

    _plot_confusion_matrix(cm, name)

    report = classification_report(
        y_test, predictions, target_names=["Legitimate (0)", "Phishing (1)"], digits=4
    )
    report_block = f"===== {name} =====\n{report}\n"

    print(
        f"  {name}: accuracy {row['accuracy']:.4f} | precision {row['precision']:.4f} "
        f"| recall {row['recall']:.4f} | F1 {row['f1']:.4f} "
        f"(prediction took {predict_time:.1f}s)"
    )
    return row, report_block


def _plot_confusion_matrix(cm, name):
    """Save one annotated confusion matrix into graphs/."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Legitimate (0)", "Phishing (1)"])
    ax.set_yticks([0, 1], labels=["Legitimate (0)", "Phishing (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {name}")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            share = 100 * cm[i, j] / total
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({share:.1f}%)",
                    ha="center", va="center", color=color)

    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GRAPHS_DIR / f"confusion_matrix_{slug(name)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


def save_reports(report_blocks):
    """Write all classification reports into one text file in outputs/."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CLASSIFICATION_REPORTS_TXT.write_text(
        "\n".join(report_blocks), encoding="utf-8"
    )
    print(f"Classification reports saved -> {config.CLASSIFICATION_REPORTS_TXT}")


def compare_models(results_df):
    """
    Purpose : Save the comparison table and draw one chart per metric.
    Input   : DataFrame with one row per model.
    Output  : outputs/model_comparison.csv + four PNG charts in graphs/.
    Return  : None
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    print(f"\nModel comparison table saved -> {config.MODEL_COMPARISON_CSV}")

    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        values = results_df[metric]
        best_idx = values.idxmax()
        colors = ["#d62728" if i == best_idx else "#1f77b4" for i in results_df.index]
        ax.bar(results_df["model"], values, color=colors)
        for i, value in zip(results_df.index, values):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(max(0.0, values.min() - 0.02), 1.005)
        ax.set_ylabel(metric.capitalize())
        title_metric = "F1 Score" if metric == "f1" else metric.capitalize()
        ax.set_title(f"Model Comparison - {title_metric} (phishing class, test set)"
                     if metric != "accuracy"
                     else "Model Comparison - Accuracy (test set)")
        ax.tick_params(axis="x", rotation=20)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", alpha=0.3)
        path = config.GRAPHS_DIR / f"model_comparison_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Graph saved -> {path}")


def identify_best_model(results_df, tuned_params):
    """
    Purpose : Pick the best model from the measured numbers and explain why
              it won, using only computed values.
    Input   : DataFrame with one row per model, tuned hyperparameters dict.
    Output  : outputs/best_model_analysis.txt
    Return  : name of the best model
    """
    ranked = results_df.sort_values(["f1", "accuracy"], ascending=False)
    best = ranked.iloc[0]
    second = ranked.iloc[1]
    worst = ranked.iloc[-1]
    test_size = int(best["tn"] + best["fp"] + best["fn"] + best["tp"])

    ranking_line = " > ".join(
        f"{r['model']} ({r['f1']:.4f})" for _, r in ranked.iterrows()
    )

    analysis = (
        f"Best model: {best['model']}\n"
        f"Selection rule: highest F1 score for the phishing class on the "
        f"held-out test set ({test_size:,} emails), with accuracy as the "
        f"tie-breaker. Tuned settings in play: Logistic Regression "
        f"C={tuned_params['lr_C']}, Naive Bayes alpha={tuned_params['nb_alpha']} "
        f"(grids and CV scores in hyperparameter_tuning.csv).\n\n"
        f"F1 ranking: {ranking_line}\n\n"
        f"{best['model']} reached accuracy {best['accuracy']:.4f}, precision "
        f"{best['precision']:.4f}, recall {best['recall']:.4f} and F1 "
        f"{best['f1']:.4f}. Out of {test_size:,} test emails it made "
        f"{int(best['fp'] + best['fn']):,} mistakes: {int(best['fp']):,} legitimate "
        f"emails flagged as phishing (false positives) and {int(best['fn']):,} "
        f"phishing emails missed (false negatives). The false negatives are the "
        f"expensive ones in this domain - each is a scam reaching an inbox - so it "
        f"matters that recall stays at {best['recall']:.4f}: the system misses "
        f"about {1000 * (1 - best['recall']):.0f} in every 1,000 phishing emails.\n\n"
        f"Why it performed best: {MODEL_REASONS[best['model']]}. "
        f"The margin over the runner-up ({second['model']}, F1 {second['f1']:.4f}) "
        f"is {best['f1'] - second['f1']:.4f}, and the gap to the weakest model "
        f"({worst['model']}, F1 {worst['f1']:.4f}) is "
        f"{best['f1'] - worst['f1']:.4f}.\n\n"
        f"Honest caution: part of what every model learns here is corpus "
        f"membership. The legitimate class draws heavily on Enron-era business "
        f"mail (Week 2 found system tokens like 'ect' among its top words even "
        f"after removing 'enron' itself), while the phishing class comes from "
        f"scam archives with their own house style. The scores measure "
        f"in-distribution separation of those sources; a mailbox with different "
        f"legitimate traffic would need retraining and fresh evaluation before "
        f"any of these numbers could be promised."
    )

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.BEST_MODEL_TXT.write_text(analysis, encoding="utf-8")
    print(f"\nBest model analysis saved -> {config.BEST_MODEL_TXT}")
    print("\n" + analysis)
    return best["model"]


def plot_feature_importance(lr_model, vectorizer, top_n=15):
    """
    Purpose : The feature-importance figure the project brief asks for:
              the words that push the decision hardest in each direction,
              read from the logistic regression coefficients.
    Input   : fitted LogisticRegression, fitted vectorizer.
    Output  : graphs/feature_importance_top_words.png
    Return  : None
    """
    features = np.array(vectorizer.get_feature_names_out())
    coef = lr_model.coef_[0]
    top_phish_idx = np.argsort(coef)[-top_n:]
    top_legit_idx = np.argsort(coef)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].barh(features[top_legit_idx][::-1], coef[top_legit_idx][::-1],
                 color="#1f77b4")
    axes[0].set_title(f"Top {top_n} legitimate-side words")
    axes[0].set_xlabel("Logistic regression coefficient")
    axes[1].barh(features[top_phish_idx], coef[top_phish_idx], color="#d62728")
    axes[1].set_title(f"Top {top_n} phishing-side words")
    axes[1].set_xlabel("Logistic regression coefficient")
    fig.suptitle("Which words drive the decision (feature importance)")
    fig.tight_layout()

    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GRAPHS_DIR / "feature_importance_top_words.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved -> {path}")
