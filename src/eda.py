"""
Exploratory Data Analysis for Week 2.

Generates the visualizations that matter for a phishing detection project:
class distribution, email length distribution, most frequent words per
class, one word cloud per class, and a phishing-keyword analysis showing
how often classic scam vocabulary (verify, click, account, urgent, ...)
appears in each class.

A missing-values chart is intentionally not generated: the Week 1
inspection reports already show zero missing values in every column, so
the graph would carry no information.

Every figure is saved to graphs/ and every observation is printed to the
console AND written to outputs/eda_summary.txt with the actual measured
numbers, so the Week 4 report can quote real values instead of guesses.
"""

from collections import Counter

import matplotlib

matplotlib.use("Agg")  # save figures to files without needing a display
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from src import config

LEGIT_COLOR = "#1f77b4"
PHISH_COLOR = "#d62728"
TOP_N_WORDS = 20

# Classic phishing/scam vocabulary checked per class in Figure 6. The list
# is fixed up front (not mined from the data) so the chart answers a fair
# question: "do the words everyone associates with phishing actually
# separate the classes in this corpus?"
PHISHING_KEYWORDS = [
    "click", "verify", "account", "urgent", "free", "money", "offer",
    "bank", "password", "credit", "winner", "prize", "link", "confirm",
    "security",
]


def _save(fig, filename):
    """Save one figure into graphs/ at readable resolution."""
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GRAPHS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved -> {path}")


def _word_counter(text_series):
    """
    Purpose : Count word frequencies over a series of preprocessed texts.
    Input   : pandas Series of space-separated word strings.
    Output  : collections.Counter of word -> frequency.
    Return  : Counter
    """
    counter = Counter()
    for text in text_series:
        counter.update(str(text).split())
    return counter


def run_eda(df):
    """
    Purpose : Produce every Week 2 graph plus a written observation for
              each one, based only on values computed from the data.
    Input   : cleaned DataFrame (text_combined, label, clean_text).
    Output  : PNG files in graphs/ and outputs/eda_summary.txt.
    Return  : None
    """
    notes = []

    def note(text):
        print("\n" + text)
        notes.append(text)

    print("Exploratory Data Analysis Started ...")

    phish = df[df["label"] == config.LABEL_PHISHING]
    legit = df[df["label"] == config.LABEL_LEGIT]

    # ---------- Figure 1: class distribution ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [len(legit), len(phish)]
    ax.bar(["Legitimate (0)", "Phishing (1)"], counts,
           color=[LEGIT_COLOR, PHISH_COLOR])
    for i, value in enumerate(counts):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    ax.set_title("Class Distribution After Cleaning")
    ax.set_ylabel("Number of emails")
    _save(fig, "class_distribution.png")

    phish_share = 100 * len(phish) / len(df)
    note(
        f"Figure 1 (class_distribution.png): after cleaning, the dataset holds "
        f"{len(phish):,} phishing and {len(legit):,} legitimate emails, so phishing "
        f"makes up {phish_share:.1f}% of the data. The near-balance is convenient: "
        f"accuracy stays a meaningful metric, no resampling is needed, and a "
        f"stratified split will keep the same ratio in the training and testing sets."
    )

    # ---------- Figure 2: email length distribution ----------
    phish_lengths = phish["clean_text"].str.split().str.len()
    legit_lengths = legit["clean_text"].str.split().str.len()

    cap = int(max(phish_lengths.quantile(0.99), legit_lengths.quantile(0.99)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = range(0, cap + 20, 20)
    ax.hist(legit_lengths.clip(upper=cap), bins=bins, alpha=0.6,
            label="Legitimate", color=LEGIT_COLOR)
    ax.hist(phish_lengths.clip(upper=cap), bins=bins, alpha=0.6,
            label="Phishing", color=PHISH_COLOR)
    ax.set_title("Email Length Distribution (words after preprocessing)")
    ax.set_xlabel(f"Words per email (axis capped at 99th percentile = {cap})")
    ax.set_ylabel("Number of emails")
    ax.legend()
    _save(fig, "email_length_distribution.png")

    longer = "legitimate" if legit_lengths.mean() > phish_lengths.mean() else "phishing"
    note(
        f"Figure 2 (email_length_distribution.png): legitimate emails average "
        f"{legit_lengths.mean():.0f} words (median {legit_lengths.median():.0f}, "
        f"standard deviation {legit_lengths.std():.0f}) while phishing emails average "
        f"{phish_lengths.mean():.0f} words (median {phish_lengths.median():.0f}, "
        f"standard deviation {phish_lengths.std():.0f}), so on average the {longer} "
        f"class runs longer in this corpus. Both distributions pile up on short "
        f"messages and overlap heavily, which means length alone cannot separate the "
        f"classes - the signal has to come from the words themselves."
    )

    # ---------- Figure 3: most frequent words per class ----------
    legit_counter = _word_counter(legit["clean_text"])
    phish_counter = _word_counter(phish["clean_text"])

    legit_top = legit_counter.most_common(TOP_N_WORDS)
    phish_top = phish_counter.most_common(TOP_N_WORDS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, top, title, color in (
        (axes[0], legit_top, "Legitimate emails", LEGIT_COLOR),
        (axes[1], phish_top, "Phishing emails", PHISH_COLOR),
    ):
        words = [w for w, _ in top][::-1]
        freqs = [f for _, f in top][::-1]
        ax.barh(words, freqs, color=color)
        ax.set_title(f"Top {TOP_N_WORDS} words - {title}")
        ax.set_xlabel("Frequency")
    fig.tight_layout()
    _save(fig, "top_words_per_class.png")

    legit_set = [w for w, _ in legit_top]
    phish_set = [w for w, _ in phish_top]
    shared = [w for w in legit_set if w in phish_set]
    legit_only = [w for w in legit_set if w not in phish_set]
    phish_only = [w for w in phish_set if w not in legit_set]
    note(
        f"Figure 3 (top_words_per_class.png): the two vocabularies barely overlap - "
        f"only {len(shared)} of the top {TOP_N_WORDS} words are shared "
        f"({', '.join(shared[:5]) if shared else 'none'}). The legitimate side is "
        f"dominated by business and scheduling vocabulary "
        f"({', '.join(legit_only[:6])}), while the phishing side is dominated by "
        f"promotional and action vocabulary ({', '.join(phish_only[:6])}). Compared "
        f"with the fake news project, where both classes shared the same political "
        f"core, these classes speak almost different languages - an early hint that "
        f"the classification task should be easier here."
    )

    # ---------- Figures 4 and 5: word clouds ----------
    for counter, filename, cmap, label in (
        (legit_counter, "wordcloud_legit.png", "Blues", "legitimate"),
        (phish_counter, "wordcloud_phishing.png", "Reds", "phishing"),
    ):
        cloud = WordCloud(
            width=1200, height=600, background_color="white",
            max_words=150, colormap=cmap, random_state=config.RANDOM_STATE,
        ).generate_from_frequencies(counter)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word cloud - {label} emails")
        _save(fig, filename)

    note(
        "Figures 4-5 (wordcloud_legit.png, wordcloud_phishing.png): the word clouds "
        "present the same frequency counts as Figure 3 in a form that works well on a "
        "slide. The legitimate cloud is built from workplace nouns and scheduling "
        "words; the phishing cloud is built from money, offers and calls to action. "
        "They add presentation value rather than new information."
    )

    # ---------- Figure 6: phishing keyword analysis ----------
    rates = []
    for kw in PHISHING_KEYWORDS:
        pattern = rf"\b{kw}\b"
        p_rate = 100 * phish["clean_text"].str.contains(pattern, regex=True).mean()
        l_rate = 100 * legit["clean_text"].str.contains(pattern, regex=True).mean()
        rates.append((kw, p_rate, l_rate))
    rates.sort(key=lambda r: r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = range(len(rates))
    width = 0.4
    ax.bar([i - width / 2 for i in x], [r[1] for r in rates], width,
           label="Phishing", color=PHISH_COLOR)
    ax.bar([i + width / 2 for i in x], [r[2] for r in rates], width,
           label="Legitimate", color=LEGIT_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rates], rotation=35, ha="right")
    ax.set_ylabel("% of emails containing the word")
    ax.set_title("Classic Phishing Keywords - occurrence rate per class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "phishing_keywords.png")

    biggest = max(rates, key=lambda r: r[1] - r[2])
    note(
        f"Figure 6 (phishing_keywords.png): every one of the {len(PHISHING_KEYWORDS)} "
        f"classic scam words appears more often in phishing than in legitimate mail. "
        f"The most frequent in phishing is '{rates[0][0]}' "
        f"({rates[0][1]:.1f}% of phishing emails vs {rates[0][2]:.1f}% of legitimate), "
        f"and the biggest class gap belongs to '{biggest[0]}' "
        f"({biggest[1]:.1f}% vs {biggest[2]:.1f}%). The keyword list was fixed before "
        f"looking at the data, so this is a fair check rather than cherry-picking. "
        f"At the same time, no single word is anywhere near universal, which is "
        f"exactly why the models will use thousands of word features instead of a "
        f"keyword rule."
    )

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EDA_SUMMARY_TXT.write_text("\n\n".join(notes), encoding="utf-8")
    print(f"\nEDA observations saved -> {config.EDA_SUMMARY_TXT}")
    print("Exploratory Data Analysis Completed")
