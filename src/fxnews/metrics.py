"""Metrics and the tests used to compare models. Everything here is plain NumPy/SciPy so it is unit-tested on a laptop."""

from __future__ import annotations

import numpy as np
from scipy import stats


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def macro_f1(y: np.ndarray, p: np.ndarray, n_classes: int = 3) -> float:
    y, p = np.asarray(y), np.asarray(p)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def confusion(y: np.ndarray, p: np.ndarray, n_classes: int = 3) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(np.asarray(y), np.asarray(p), strict=True):
        m[a, b] += 1
    return m


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over test examples for any metric fn(y, p)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(y), len(y))
        vals[i] = fn(y[idx], p[idx])
    return float(fn(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test: do models a and b differ in accuracy on the same examples? Returns (a-only correct, b-only correct, p)."""
    ok_a, ok_b = np.asarray(a) == np.asarray(y), np.asarray(b) == np.asarray(y)
    n10, n01 = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, float(min(1.0, stats.binomtest(n10, n10 + n01, 0.5).pvalue))


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj.tolist()
