# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2026-07-03

First complete release: the full internship project across its four phases.

### Phase 1 — Data pipeline (Week 1)
- Modular project structure (`src/` + single `main.py` entry point, seed 42).
- Loader with column validation; before/after inspection reports with
  measured content markers (uppercase, HTML, URL tokens, digits).
- Cleaning: 20,000-character truncation of extreme outliers (run before
  deduplication, on purpose), 409 duplicates removed, label-conflict check
  (0 found), `enron` source watermark measured (18.2% vs 0.0%) and made a
  stop word. Final corpus: 82,075 emails.

### Phase 2 — EDA & feature engineering (Week 2)
- Six figures with written, number-backed observations (class balance,
  length distributions, top words, word clouds, fixed-list phishing-keyword
  analysis).
- Stratified 80/20 split before any vectorizer fitting (65,660 / 16,415).
- CountVectorizer vs TF-IDF decided by a pre-declared cross-validated
  protocol; statistical tie (gap 0.0003) resolved to TF-IDF by the declared
  tie rule.

### Phase 3 — Models, tuning & evaluation (Week 3)
- Four models trained on identical features: Logistic Regression,
  Multinomial Naive Bayes, Random Forest, MLP.
- Scoped GridSearchCV (LR `C` -> 10, CV F1 0.9801 -> 0.9827; NB `alpha` ->
  0.1), full search recorded in `outputs/hyperparameter_tuning.csv`.
- Per-model metrics, confusion matrices, comparison charts; best model
  (tuned Logistic Regression: accuracy 0.9822, recall 0.9851, F1 0.9829)
  selected automatically and saved with the fitted vectorizer.
- Coefficient-level feature-importance figure.

### Phase 4 — Documentation (Week 4)
- 4-page IEEE conference paper (IEEEtran) written exclusively from the
  pipeline's outputs, including the archive-watermark analysis.
- 11-page project report (DOCX + PDF) with 9 embedded figures.
- 6-slide presentation (PPTX + PDF) with speaker notes on every slide,
  plus `presentation/speaker_notes.md` extracted from the deck itself.

### Repository packaging
- Professional README with gallery, MIT license, contribution guidelines,
  code of conduct, security and support policies, changelog, `.gitignore`,
  and an `assets/` folder with byte-identical copies of the real figures.
