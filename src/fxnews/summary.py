"""Turn predictions into the results table: point estimates with bootstrap intervals, per-class F1, paired tests."""

from __future__ import annotations

import numpy as np

from fxnews import metrics
from fxnews.data import LABELS


def per_class_f1(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {}
    for c, name in enumerate(LABELS):
        tp, fp, fn = np.sum((y == c) & (p == c)), np.sum((y != c) & (p == c)), np.sum((y == c) & (p != c))
        out[name] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return out


def summarise(y: np.ndarray, pred: np.ndarray, ms_per_example: float | None = None, reps: int = 2000) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "accuracy": metrics.bootstrap_ci(y, pred, metrics.accuracy, reps),
        "macro_f1": metrics.bootstrap_ci(y, pred, metrics.macro_f1, reps),
        "per_class_f1": per_class_f1(y, pred),
        "confusion": metrics.confusion(y, pred).tolist(),
        "ms_per_example": ms_per_example,
    }


def compare(y: np.ndarray, preds: dict[str, np.ndarray], reference: str) -> dict[str, dict]:
    """Paired exact McNemar of every model against `reference`, Holm-adjusted across the comparisons."""
    names = [n for n in preds if n != reference]
    raw = [metrics.mcnemar(y, preds[n], preds[reference]) for n in names]  # a = model, b = reference
    adj = metrics.holm([r[2] for r in raw])
    return {n: {"model_only_correct": r[0], "reference_only_correct": r[1], "p": r[2], "p_holm": a} for n, r, a in zip(names, raw, adj, strict=True)}


def markdown_table(results: dict[str, dict], extras: dict[str, dict[str, str]] | None = None) -> str:
    """`results`: name -> summarise() output. `extras`: name -> {column: text} (e.g. training time, GPU memory)."""
    extra_cols = sorted({k for v in (extras or {}).values() for k in v})
    head = ["model", "accuracy (95% CI)", "macro-F1 (95% CI)", "ms / example", *extra_cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in results.items():
        a, f = r["accuracy"], r["macro_f1"]
        ms = "" if r["ms_per_example"] is None else f"{r['ms_per_example']:.1f}"
        row = [name, f"{a[0]:.3f} ({a[1]:.3f}-{a[2]:.3f})", f"{f[0]:.3f} ({f[1]:.3f}-{f[2]:.3f})", ms, *[(extras or {}).get(name, {}).get(c, "") for c in extra_cols]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
