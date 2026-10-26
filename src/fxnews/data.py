"""Data for the sentiment task and the split protocol.

Task: classify a short financial-news headline as Bearish, Bullish or Neutral for the asset it mentions.
Dataset: `zeroshot/twitter-financial-news-sentiment` on the Hugging Face Hub (about 11.9k labelled tweets
from financial news accounts; labels 0 Bearish, 1 Bullish, 2 Neutral). The Hub page states the licence; check it
before any use beyond research, and it is downloaded by the notebook, never stored in this repository.

Protocol (fixed before any model is trained):
  * the dataset's own `validation` split is the TEST set, touched once at the end;
  * `train` is split 90/10 (stratified, seeded) into fit and DEV, and DEV is the only place anything is tuned;
  * any text that appears in both train and test (exact match after normalisation) is removed from train, and
    the number removed is reported, because duplicated headlines inflate every model's score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

LABELS = ["Bearish", "Bullish", "Neutral"]
HUB_NAME = "zeroshot/twitter-financial-news-sentiment"
SEED = 42


def normalise(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%.' ]+", " ", text)).strip()


@dataclass
class Splits:
    fit_x: list[str]
    fit_y: np.ndarray
    dev_x: list[str]
    dev_y: np.ndarray
    test_x: list[str]
    test_y: np.ndarray
    removed_duplicates: int


def make_splits(train_x: list[str], train_y: list[int], test_x: list[str], test_y: list[int], dev_frac: float = 0.1, seed: int = SEED) -> Splits:
    ty = np.asarray(train_y)
    test_norm = {normalise(t) for t in test_x}
    keep = np.array([normalise(t) not in test_norm for t in train_x])
    removed = int((~keep).sum())
    tx = [t for t, k in zip(train_x, keep, strict=True) if k]
    ty = ty[keep]
    rng = np.random.default_rng(seed)
    dev_idx = []
    for c in np.unique(ty):  # stratified: every class keeps the same share in DEV
        idx = np.flatnonzero(ty == c)
        rng.shuffle(idx)
        dev_idx += idx[: max(1, int(round(len(idx) * dev_frac)))].tolist()
    mask = np.zeros(len(tx), dtype=bool)
    mask[dev_idx] = True
    return Splits([t for t, m in zip(tx, mask, strict=True) if not m], ty[~mask], [t for t, m in zip(tx, mask, strict=True) if m], ty[mask],
                  list(test_x), np.asarray(test_y), removed)


def load_hub():
    """Returns (train_x, train_y, test_x, test_y). Needs the `datasets` package and network access."""
    from datasets import load_dataset

    ds = load_dataset(HUB_NAME)
    return ds["train"]["text"], ds["train"]["label"], ds["validation"]["text"], ds["validation"]["label"]
