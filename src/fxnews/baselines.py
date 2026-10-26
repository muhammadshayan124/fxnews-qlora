"""The non-LLM baselines every fine-tuned model has to be compared with.

* Rules: a hand-written keyword lexicon. It is the "could a few lines of code do this" baseline. Written by the
  repo author before looking at any model output; the word lists are short on purpose.
* TF-IDF + logistic regression: the strong classical baseline. The regularisation strength is chosen on DEV only.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BEARISH = {"falls", "fall", "drops", "drop", "slump", "slumps", "plunge", "plunges", "tumble", "tumbles", "sinks", "cuts", "cut", "misses", "miss",
           "downgrade", "downgrades", "downgraded", "lower", "weak", "warns", "warning", "loss", "losses", "lawsuit", "probe", "recall", "bankruptcy",
           "layoffs", "decline", "declines", "sell", "bearish", "short"}
BULLISH = {"rises", "rise", "jumps", "jump", "surges", "surge", "soars", "soar", "rallies", "rally", "gains", "gain", "beats", "beat", "upgrade",
           "upgrades", "upgraded", "higher", "strong", "record", "raises", "raise", "profit", "wins", "approval", "approved", "buy", "bullish", "growth"}


def rule_predict(text: str) -> int:
    """0 Bearish, 1 Bullish, 2 Neutral: whichever lexicon has more hits, Neutral on a tie or no hits."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    bear, bull = len(words & BEARISH), len(words & BULLISH)
    return 0 if bear > bull else 1 if bull > bear else 2


def majority(fit_y, n: int) -> np.ndarray:
    """Always predict the most frequent class in the fit split. The floor that any model must clear to be saying anything."""
    return np.full(n, int(np.bincount(np.asarray(fit_y)).argmax()))


def rules(texts: list[str]) -> np.ndarray:
    return np.array([rule_predict(t) for t in texts])


def fit_tfidf(fit_x: list[str], fit_y, dev_x: list[str], dev_y, grid=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
    """Choose C on DEV (macro-F1), then refit on fit+dev. Returns (model, chosen C, DEV score per C)."""
    from fxnews.metrics import macro_f1

    scores = {}
    for c in grid:
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=c, max_iter=2000))
        m.fit(fit_x, fit_y)
        scores[c] = macro_f1(dev_y, m.predict(dev_x))
    best = max(scores, key=scores.get)
    final = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=best, max_iter=2000))
    final.fit(list(fit_x) + list(dev_x), np.concatenate([np.asarray(fit_y), np.asarray(dev_y)]))
    return final, best, scores
