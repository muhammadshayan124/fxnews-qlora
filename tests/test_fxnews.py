from __future__ import annotations

import numpy as np
import pytest

from fxnews import baselines, data, llm, metrics


class FakeTok:
    """Word-level stand-in tokenizer: stable ids, eos = 0."""

    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=False, **kw):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]
        return {"input_ids": ids}


def test_normalise_and_duplicate_removal():
    assert data.normalise("AAPL jumps 5%! https://t.co/xyz") == "aapl jumps 5%"
    tx = ["AAPL jumps 5%! https://t.co/a", "Oil falls", "Gold steady", "Tesla recall widens"]
    ty = [1, 0, 2, 0]
    s = data.make_splits(tx, ty, ["aapl jumps 5%", "Brand new headline"], [1, 2], dev_frac=0.34)
    assert s.removed_duplicates == 1
    all_train = s.fit_x + s.dev_x
    assert "AAPL jumps 5%! https://t.co/a" not in all_train and len(all_train) == 3
    assert set(s.test_x) == {"aapl jumps 5%", "Brand new headline"}


def test_split_is_stratified_seeded_and_disjoint():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    x = [f"headline number {i}" for i in range(600)]
    a = data.make_splits(x, list(y), ["t1"], [0])
    b = data.make_splits(x, list(y), ["t1"], [0])
    assert a.dev_x == b.dev_x  # deterministic
    assert set(a.fit_x).isdisjoint(a.dev_x) and len(a.fit_x) + len(a.dev_x) == 600
    shares = [np.mean(a.dev_y == c) for c in range(3)]
    assert max(shares) - min(shares) < 0.1 and abs(len(a.dev_x) - 60) <= 3


def test_metrics_hand_checked():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    assert metrics.accuracy(y, p) == pytest.approx(4 / 6)
    assert metrics.macro_f1(y, p) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert metrics.confusion(y, p).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    est, lo, hi = metrics.bootstrap_ci(y, p, metrics.accuracy, reps=500)
    assert lo <= est <= hi
    assert metrics.macro_f1(y, y) == 1.0


def test_mcnemar_and_holm():
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)
    b = np.concatenate([np.ones(9, dtype=int), np.zeros(11, dtype=int)])
    n10, n01, p = metrics.mcnemar(y, a, b)
    assert (n10, n01) == (9, 0) and p < 0.01
    assert metrics.mcnemar(y, a, a)[2] == 1.0
    assert metrics.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_majority_baseline_predicts_the_most_frequent_class():
    assert baselines.majority([0, 2, 2, 1, 2], 4).tolist() == [2, 2, 2, 2]


def test_rules_baseline():
    assert baselines.rule_predict("Shares surge after earnings beat") == 1
    assert baselines.rule_predict("Company warns of weak demand, cuts outlook") == 0
    assert baselines.rule_predict("Annual meeting scheduled for May") == 2


def test_tfidf_learns_a_separable_toy_problem():
    bull = [f"stock surges record profit number {i}" for i in range(30)]
    bear = [f"stock plunges heavy loss number {i}" for i in range(30)]
    neut = [f"company announces meeting date number {i}" for i in range(30)]
    x, y = bull + bear + neut, [1] * 30 + [0] * 30 + [2] * 30
    model, c, scores = baselines.fit_tfidf(x[::2], y[::2], x[1::2], y[1::2])
    assert c in scores and max(scores.values()) > 0.9
    assert list(model.predict(["stock surges on record profit", "stock plunges on heavy loss"])) == [1, 0]


def test_prompt_shots_and_balanced_selection():
    p = llm.build_prompt("Oil falls", [("Gold rises", 1)])
    assert p.endswith("Headline: Oil falls\nSentiment:") and "Headline: Gold rises\nSentiment: Bullish" in p
    texts = [f"t{i}" for i in range(90)]
    labels = np.array([i % 3 for i in range(90)])
    shots = llm.choose_shots(texts, labels, (3, 3, 2), seed=1)
    assert sorted(y for _, y in shots) == [0, 0, 0, 1, 1, 1, 2, 2]
    assert shots == llm.choose_shots(texts, labels, (3, 3, 2), seed=1)


def test_sft_example_masks_the_prompt_and_learns_only_the_answer():
    tok = FakeTok()
    ex = llm.make_sft_example(tok, "Oil falls sharply", 0)
    n_prompt = sum(1 for x in ex["labels"] if x == -100)
    assert ex["labels"][:n_prompt] == [-100] * n_prompt and ex["input_ids"][:n_prompt] == tok(llm.build_prompt("Oil falls sharply"))["input_ids"]
    assert ex["labels"][n_prompt:] == [tok.vocab["Bearish"], tok.eos_token_id]  # the label word, then end of sequence
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_collate_pads_inputs_attention_and_labels_consistently():
    batch = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [-100]}]
    out = llm.collate(batch, pad_id=9)
    assert out["input_ids"] == [[5, 6, 7], [5, 9, 9]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"] == [[-100, 6, 7], [-100, -100, -100]]


def test_label_tokens_must_be_distinguishable():
    tok = FakeTok()
    assert len(set(llm.label_token_ids(tok))) == 3

    class Collide(FakeTok):
        def __call__(self, text, add_special_tokens=False, **kw):
            return {"input_ids": [1, 2]}

    with pytest.raises(ValueError, match="share a first token"):
        llm.label_token_ids(Collide())


def test_summary_and_comparison_and_table():
    from fxnews import summary

    y = np.array([0, 1, 2] * 20)
    good = y.copy()
    bad = y.copy()
    bad[:20] = (bad[:20] + 1) % 3
    s = summary.summarise(y, good, 1.5, reps=200)
    assert s["accuracy"][0] == 1.0 and s["per_class_f1"]["Bullish"] == 1.0 and s["ms_per_example"] == 1.5
    cmp = summary.compare(y, {"good": good, "bad": bad, "ref": bad}, "ref")
    assert cmp["good"]["model_only_correct"] > 0 and cmp["good"]["p_holm"] >= cmp["good"]["p"]
    assert cmp["bad"]["p"] == 1.0
    md = summary.markdown_table({"good": s, "bad": summary.summarise(y, bad, None, reps=200)}, {"good": {"train min": "12"}})
    assert md.splitlines()[0].startswith("| model |") and "| good | 1.000" in md and "train min" in md
