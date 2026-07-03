"""
Model training for Week 3.

The Project 2 brief asks for Logistic Regression, Naive Bayes, Random
Forest and a simple Neural Network, trained on the TF-IDF features that
Week 2 selected, plus hyperparameter tuning. All four models share the
identical training matrix and are scored on the identical test set.

Tuning scope - and why it is scoped. GridSearchCV (3-fold, training set
only, F1 on the phishing class) searches the regularisation strength C
for Logistic Regression and the smoothing alpha for Naive Bayes: these
grids cost seconds and genuinely move the needle. The Random Forest stays
at 100 trees (forests are famously insensitive to adding trees beyond
this point - it mainly buys training time - and 100 keeps results
comparable with Project 1) and the MLP keeps one hidden layer of 100
units with early stopping, which self-regularises by construction. The
searched grids, candidates and winners are written to
outputs/hyperparameter_tuning.csv so the scope of the search is visible
rather than implied.

Every model that accepts random_state gets the project seed, so reruns
reproduce identical numbers.
"""

import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src import config


def tune_models(X_train, y_train):
    """
    Purpose : Grid-search the compute-cheap hyperparameters on the
              training set only (3-fold CV, F1 on the phishing class).
    Input   : training matrix and labels.
    Output  : outputs/hyperparameter_tuning.csv
    Return  : dict with the winning lr_C and nb_alpha.
    """
    print("Hyperparameter Tuning Started (3-fold CV on training set) ...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []

    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        {"C": [0.1, 1.0, 10.0]}, cv=cv, scoring="f1", n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    rows.append({
        "model": "Logistic Regression", "parameter": "C",
        "candidates": "0.1 / 1.0 / 10.0",
        "best_value": lr_grid.best_params_["C"],
        "best_cv_f1": lr_grid.best_score_,
    })
    print(f"  Logistic Regression : best C = {lr_grid.best_params_['C']} "
          f"(CV F1 {lr_grid.best_score_:.4f})")

    nb_grid = GridSearchCV(
        MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]},
        cv=cv, scoring="f1", n_jobs=-1,
    )
    nb_grid.fit(X_train, y_train)
    rows.append({
        "model": "Multinomial Naive Bayes", "parameter": "alpha",
        "candidates": "0.1 / 0.5 / 1.0",
        "best_value": nb_grid.best_params_["alpha"],
        "best_cv_f1": nb_grid.best_score_,
    })
    print(f"  Multinomial NB      : best alpha = {nb_grid.best_params_['alpha']} "
          f"(CV F1 {nb_grid.best_score_:.4f})")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.TUNING_CSV, index=False)
    print(f"Tuning table saved -> {config.TUNING_CSV}")

    return {"lr_C": lr_grid.best_params_["C"],
            "nb_alpha": nb_grid.best_params_["alpha"]}


def get_models(lr_C=1.0, nb_alpha=1.0):
    """
    Purpose : Build the four models required by the Project 2 brief,
              using the tuned hyperparameters where applicable.
    Input   : lr_C, nb_alpha from tune_models().
    Output  : dict of display name -> unfitted estimator.
    Return  : dict
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=lr_C, max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=nb_alpha),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=config.RANDOM_STATE,
        ),
    }


def slug(name):
    """Turn a display name into a safe file name part."""
    return name.lower().replace(" ", "_")


def train_model(name, model, X_train, y_train):
    """
    Purpose : Fit one model on the training features and save it.
    Input   : display name, unfitted estimator, training matrix + labels.
    Output  : models/<slug>.joblib on disk.
    Return  : (fitted model, training time in seconds)
    """
    print(f"\nTraining {name} ...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{slug(name)}.joblib"
    joblib.dump(model, path, compress=3)

    print(f"  Trained in {train_time:.1f}s | saved -> {path}")
    return model, train_time
