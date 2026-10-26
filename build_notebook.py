"""Generate notebooks/fxnews_qlora.ipynb.

The notebook embeds the library files from src/fxnews as `%%writefile` cells, so the code that runs on Colab is
byte-for-byte the code the unit tests exercise locally. Regenerate after editing the library:  python build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
SRC = ROOT / "src" / "fxnews"
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# QLoRA fine-tuning for financial-news sentiment: does it beat the simple baselines?

A 1.5B-parameter model is fine-tuned with 4-bit QLoRA on a labelled set of financial-news headlines
(Bearish / Bullish / Neutral) and compared **on the same held-out test set** against:

0. the **majority class** (the test set is about two-thirds Neutral, so this is the floor),
1. a hand-written keyword **rules** baseline,
2. **TF-IDF + logistic regression** (the strong classical baseline),
3. the **same base model, zero-shot**,
4. the **same base model, few-shot** (8 demonstrations).

Everything is scored the same way (next-token probabilities over the three label words), with bootstrap
confidence intervals and paired exact McNemar tests (Holm-adjusted). Nothing is tuned on the test set.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then *Run all*. Expected time: about 35-50 minutes
(training is about 15-25 of those). The last cell writes `fxnews_results.zip`; download it and keep it, the
repository's README is filled from `results.json` inside it.
""")

md("## 1. Setup")
code("""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46,<5", "peft>=0.13", "bitsandbytes>=0.44", "accelerate>=1.0",
                "datasets>=3.0", "scikit-learn", "scipy", "matplotlib", "pandas"], check=True)
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU"
print(torch.__version__, torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")

md("## 2. The library (identical to the unit-tested source in `src/fxnews`)")
code("import os\nos.makedirs('fxnews', exist_ok=True)")
for fname in ("__init__.py", "data.py", "metrics.py", "baselines.py", "summary.py", "llm.py"):
    body = (SRC / fname).read_text(encoding="utf-8")
    code(f"%%writefile fxnews/{fname}\n{body}")

md("""
## 3. Configuration
All choices are made **here, once, before looking at any result**.
""")
code("""
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache-2.0. A 3B model also fits on a T4: "Qwen/Qwen2.5-3B-Instruct"
EPOCHS, LR, LORA_R, LORA_ALPHA = 2, 2e-4, 16, 32
SEED = 42
import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
""")

md("""
## 4. Data and split protocol
The dataset's own `validation` split is the **test set** and is used once, at the end. `train` is split 90/10 into
*fit* and *dev*; only dev is used for choices. Headlines that appear in both train and test are removed from train
(the number is reported).
""")
code("""
import numpy as np, pandas as pd
from fxnews import data
tx, ty, vx, vy = data.load_hub()
S = data.make_splits(tx, ty, vx, vy)
print(f"fit {len(S.fit_x)} | dev {len(S.dev_x)} | test {len(S.test_x)} | duplicates removed from train: {S.removed_duplicates}")
print(pd.DataFrame({"fit": np.bincount(S.fit_y), "dev": np.bincount(S.dev_y), "test": np.bincount(S.test_y)}, index=data.LABELS))
""")

md("## 5. Baselines: rules and TF-IDF + logistic regression (CPU)")
code("""
import time
from fxnews import baselines, summary
results, preds, extras, probs = {}, {}, {}, {}

preds["majority class"] = baselines.majority(S.fit_y, len(S.test_x))
results["majority class"] = summary.summarise(S.test_y, preds["majority class"], 0.0)
preds["rules"] = baselines.rules(S.test_x)
results["rules"] = summary.summarise(S.test_y, preds["rules"], 0.0)

tfidf, C, dev_scores = baselines.fit_tfidf(S.fit_x, S.fit_y, S.dev_x, S.dev_y)
t0 = time.perf_counter(); preds["tfidf+logreg"] = tfidf.predict(S.test_x)
results["tfidf+logreg"] = summary.summarise(S.test_y, preds["tfidf+logreg"], (time.perf_counter() - t0) * 1000 / len(S.test_x))
print("TF-IDF C chosen on dev:", C, {k: round(v, 3) for k, v in dev_scores.items()})
for k in ("majority class", "rules", "tfidf+logreg"):
    print(k, results[k]["accuracy"][0], results[k]["macro_f1"][0])
""")

md("""
## 6. The base model: zero-shot and few-shot
The base model is loaded once in 4-bit. Both prompting variants read the label from the next-token probabilities.
The 8 demonstrations come from the *fit* split (class-balanced, fixed seed).
""")
code("""
from fxnews import llm
model, tok = llm.load_model(MODEL, four_bit=True)
print("label first-token ids:", llm.label_token_ids(tok))

p, pred, ms = llm.score(model, tok, S.test_x);               preds["zero-shot"] = pred; probs["zero-shot"] = p
results["zero-shot"] = summary.summarise(S.test_y, pred, ms)
SHOTS = llm.choose_shots(S.fit_x, S.fit_y, (3, 3, 2), seed=SEED)
p, pred, ms = llm.score(model, tok, S.test_x, shots=SHOTS);  preds["few-shot (8)"] = pred; probs["few-shot (8)"] = p
results["few-shot (8)"] = summary.summarise(S.test_y, pred, ms)
for k in ("zero-shot", "few-shot (8)"):
    print(k, round(results[k]["accuracy"][0], 3), round(results[k]["macro_f1"][0], 3), f"{results[k]['ms_per_example']:.1f} ms/ex")
""")

md("## 7. QLoRA fine-tuning")
code("""
import time, torch
model = llm.add_lora(model, r=LORA_R, alpha=LORA_ALPHA)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad); total = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {trainable/1e6:.1f}M of {total/1e6:.0f}M ({100*trainable/total:.2f}%)")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer = llm.train_lora(model, tok, S.fit_x, S.fit_y, S.dev_x, S.dev_y, "qlora_out", epochs=EPOCHS, lr=LR, seed=SEED)
train_minutes = (time.time() - t0) / 60
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"training: {train_minutes:.1f} min, peak GPU memory {peak_gb:.1f} GB")
""")
code("""
import matplotlib.pyplot as plt
log = trainer.state.log_history
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot([l["step"] for l in log if "loss" in l], [l["loss"] for l in log if "loss" in l], label="train loss")
ev = [(l["step"], l["eval_loss"]) for l in log if "eval_loss" in l]
if ev: ax.plot(*zip(*ev), marker="o", label="dev loss")
ax.set_xlabel("step"); ax.legend(); fig.tight_layout(); fig.savefig("training_curve.png", dpi=130); plt.show()
""")

md("## 8. Final evaluation on the test set (used once)")
code("""
p, pred, ms = llm.score(model, tok, S.test_x);  preds["QLoRA fine-tuned"] = pred; probs["QLoRA fine-tuned"] = p
results["QLoRA fine-tuned"] = summary.summarise(S.test_y, pred, ms)
extras = {"QLoRA fine-tuned": {"train min": f"{train_minutes:.0f}", "peak GPU GB": f"{peak_gb:.1f}", "trainable params": f"{trainable/1e6:.1f}M"}}
order = ["majority class", "rules", "tfidf+logreg", "zero-shot", "few-shot (8)", "QLoRA fine-tuned"]
table = summary.markdown_table({k: results[k] for k in order}, extras)
print(table)
""")
code("""
cmp = summary.compare(S.test_y, {k: preds[k] for k in order}, reference="tfidf+logreg")
print("Paired exact McNemar versus TF-IDF + logistic regression (Holm-adjusted):")
for k, v in cmp.items():
    print(f"  {k:18s} model-only-correct {v['model_only_correct']:4d} | tfidf-only-correct {v['reference_only_correct']:4d} | p {v['p']:.4f} | p_holm {v['p_holm']:.4f}")
cmp_ft = summary.compare(S.test_y, {k: preds[k] for k in ("few-shot (8)", "QLoRA fine-tuned")}, reference="few-shot (8)")
print("Fine-tuned vs few-shot:", cmp_ft)
for k in ("tfidf+logreg", "few-shot (8)", "QLoRA fine-tuned"):
    print(k, "per-class F1:", {a: round(b, 3) for a, b in results[k]["per_class_f1"].items()})
""")

md("## 9. Error analysis: what does the fine-tuned model still get wrong?")
code("""
import numpy as np
wrong = np.flatnonzero(preds["QLoRA fine-tuned"] != S.test_y)
rng = np.random.default_rng(0)
for i in rng.choice(wrong, min(15, len(wrong)), replace=False):
    pr = probs["QLoRA fine-tuned"][i]
    print(f"gold {data.LABELS[S.test_y[i]]:8s} | pred {data.LABELS[preds['QLoRA fine-tuned'][i]]:8s} ({pr.max():.2f}) | {S.test_x[i][:140]}")
print(f"\\n{len(wrong)} errors of {len(S.test_y)}")
""")

md("""
## 10. Optional ablation (off by default; about 20 minutes each)
Set `RUN_ABLATION = True` to compare LoRA rank 4 against 16 on **dev** (never test). It reloads the base model, so
run it last.
""")
code("""
RUN_ABLATION = False
if RUN_ABLATION:
    import gc
    ablation = {}
    for r in (4, 16):
        del model; gc.collect(); torch.cuda.empty_cache()
        m, t = llm.load_model(MODEL, four_bit=True)
        m = llm.add_lora(m, r=r, alpha=2 * r)
        llm.train_lora(m, t, S.fit_x, S.fit_y, S.dev_x, S.dev_y, f"abl_r{r}", epochs=EPOCHS, lr=LR, seed=SEED)
        _, pd_, _ = llm.score(m, t, S.dev_x)
        from fxnews import metrics
        ablation[f"r={r}"] = metrics.macro_f1(S.dev_y, pd_)
        model = m
    print("dev macro-F1:", ablation)
""")

md("## 11. Save everything")
code("""
import json, shutil, platform
out = {
    "model": MODEL, "seed": SEED, "epochs": EPOCHS, "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "n": {"fit": len(S.fit_x), "dev": len(S.dev_x), "test": len(S.test_x), "duplicates_removed": S.removed_duplicates},
    "results": results, "comparisons_vs_tfidf": cmp, "table_markdown": table,
    "predictions": {k: [int(x) for x in v] for k, v in preds.items()}, "test_labels": [int(x) for x in S.test_y],
}
json.dump(out, open("results.json", "w"), indent=2)
model.save_pretrained("qlora_adapter")           # the adapter only (tens of MB), not the base model
shutil.make_archive("fxnews_results", "zip", ".", "qlora_adapter")
import zipfile
with zipfile.ZipFile("fxnews_results.zip", "a") as z:
    for f in ("results.json", "training_curve.png"): z.write(f)
print("wrote fxnews_results.zip  ->  download it from the Files panel")
try:
    from google.colab import files; files.download("fxnews_results.zip")
except Exception:
    pass
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = ROOT / "notebooks" / "fxnews_qlora.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
