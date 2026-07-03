# Speaker Notes — Phishing Email Detection (IICT, 6 slides)

Extracted from the .pptx notes (View > Notes Page). Rough timing:
about 1.5–2 minutes per slide, 10–12 minutes total.

## Slide 1 — Problem Statement

Good morning everyone. Project 2 of the IICT internship: detecting phishing emails from text alone. The question on the slide has two halves. The first is the obvious one — can a model catch scams. The second is the one this project takes seriously — how much of a high score is genuine phishing signal, and how much is the model recognising which archive an email came from. Three headline figures: about eighty-two thousand emails, four models required by the brief, all tuned and compared under identical conditions, and the winner catches 98.5 percent of phishing — recall is the metric I will keep coming back to, because a missed scam costs far more than a false alarm.

## Slide 2 — Dataset

The corpus combines Enron-era business mail and mailing lists on the legitimate side with public scam archives on the phishing side — 52/48, so no resampling needed. The habit this project runs on: measure before you trust. The authors say the text is pre-normalised — I checked: zero percent uppercase, zero percent HTML, but nearly half the emails still carry URL tokens, so preprocessing still matters. The word 'enron' appears in eighteen percent of legitimate mail and zero percent of phishing — that is a label for where the ham was collected, not a scam signal, so it became a stop word. One subtle ordering bug worth mentioning: truncation runs before de-duplication, because two long emails that differ only beyond the cap become identical after it. And the chart: fifteen classic scam words chosen before looking at the data — every single one is more frequent in phishing, but none is universal, which is why we use five thousand word features, not a keyword rule.

## Slide 3 — Methodology

One rule above all: split before fitting anything. The vectorizer only ever sees training text. For the representation I ran counts against TF-IDF with an identical logistic-regression probe under three-fold cross-validation — and got a statistical tie, a gap of three ten-thousandths, inside the tie threshold I declared before running the comparison. The pre-declared rule breaks ties toward TF-IDF, which down-weights near-universal words and is what the brief suggests. Why bother comparing at all, then? Because in Project 1 the identical protocol produced a decisive win for raw counts. Same method, different data, different answer — that is exactly the point of measuring instead of defaulting. The fourth card matters too: hyperparameters are grid-searched where the search is cheap and meaningful, and the scope of that search is written to a file rather than left implied.

## Slide 4 — Models Used

The four models the brief requires, all on identical matrices and the identical test set — any difference comes from the algorithm, not the setup. The amber-T models are grid-search tuned: logistic regression's C, searched over point-one, one, and ten, landed on ten and lifted cross-validated F1 from 0.9801 to 0.9827. Naive Bayes's smoothing landed on point-one, beating the default. The navy-F models are fixed with the reasoning written down: forests barely change beyond a hundred trees, so the compute budget went where it moves the needle, and the early-stopped MLP regularises itself. Everything — grids, candidates, winners — is saved to a tuning CSV by the pipeline, so the scope of the search is visible rather than implied. All four trained models are saved with joblib.

## Slide 5 — Results

Everything on one chart. Tuned logistic regression wins: 98.22 percent accuracy, F1 0.9829, and — the number I care most about — recall 0.9851. Out of sixteen and a half thousand test emails it makes 293 mistakes: 165 false alarms and 128 missed phish, roughly fifteen misses per thousand phishing emails. Three things to notice. The top three models sit within eight ten-thousandths of F1 — with vocabularies this disjoint the classes are nearly linearly separable, so the MLP's extra capacity re-draws the same boundary at a hundred times the training cost, and the forest trades recall for the best precision. Naive Bayes is the cautionary row: decent accuracy, but 498 missed phishing emails — four times the winner's — which is disqualifying in this domain. And the error pattern at the bottom: the misses skew towards long emails that blend business vocabulary; the short punchy scams are caught almost perfectly.

## Slide 6 — Conclusion

To close: a pipeline where every decision has a measurement behind it — the pre-normalisation claim verified, the watermark quantified before removal, the representation chosen by a declared protocol, tuning scoped and recorded. Tuned logistic regression wins with the best F1 and the best recall — 128 misses out of eight and a half thousand phishing emails. The finding I would defend hardest is the last card. Reading the model's strongest weights: alongside genuine scam vocabulary like meds, replica and kindly, the single strongest phishing feature is josemonkeyorg — the hostname of a public phishing archive — and the legitimate side contains a SpamAssassin token and the first names of Enron executives. The model is partly a phishing detector and partly an archive detector. That is what near-perfect accuracy means on a combined corpus, and it is why cross-archive evaluation tops the future-work list. Thank you — happy to take questions.
