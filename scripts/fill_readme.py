"""Fill the README results table from the notebook's results.json.

Usage: python scripts/fill_readme.py path/to/results.json
Replaces everything between <!-- results:start --> and <!-- results:end -->, and writes the paired comparisons below it.
"""

import json
import sys
from pathlib import Path

res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
readme = Path(__file__).resolve().parents[1] / "README.md"
text = readme.read_text(encoding="utf-8")

lines = [res["table_markdown"], ""]
lines.append(f"Setup: `{res['model']}`, LoRA r={res['lora_r']}, {res['epochs']} epochs, lr {res['lr']}, seed {res['seed']}, {res['gpu']}. "
             f"fit {res['n']['fit']} / dev {res['n']['dev']} / test {res['n']['test']}.")
lines += ["", "Paired exact McNemar versus TF-IDF + logistic regression (Holm-adjusted across comparisons):", "",
          "| model | model-only correct | TF-IDF-only correct | p | p (Holm) |", "|---|---|---|---|---|"]
for k, v in res["comparisons_vs_tfidf"].items():
    lines.append(f"| {k} | {v['model_only_correct']} | {v['reference_only_correct']} | {v['p']:.1e} | {v['p_holm']:.1e} |")
body = "\n".join(lines)

a, b = "<!-- results:start -->", "<!-- results:end -->"
i, j = text.find(a), text.find(b)
if i == -1 or j < i:
    raise SystemExit("README markers not found")
readme.write_text(text[: i + len(a)] + "\n" + body + "\n" + text[j:], encoding="utf-8")
print("updated", readme)
