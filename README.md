# fxnews-qlora

Does fine-tuning a small language model with QLoRA beat the simple alternatives, on a real classification
task, under an evaluation that does not flatter anyone?

The task is financial-news sentiment: label a short headline **Bearish / Bullish / Neutral**. A 1.5B-parameter
model (`Qwen2.5-1.5B-Instruct`, 4-bit, LoRA rank 16) is fine-tuned on a T4 GPU and compared on one held-out
test set against a majority-class floor, a keyword-rules baseline, TF-IDF + logistic regression, and the *same*
base model prompted zero-shot and few-shot.

> **Status.** The data pipeline, baselines, metrics, statistics and the Colab notebook are unit-tested (12 tests), and the notebook has been run end to end on a
> Colab T4 (single run, seed 42). All numbers below come from that run (`results/colab_results.json`, training curve in `results/training_curve.png`). The LoRA adapter
> (74 MB) is not committed; it belongs on the Hugging Face Hub.

## Data and protocol

Dataset: [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
(Hugging Face Hub; check its licence before any use beyond research; it is downloaded by the code, not stored here).
9,543 training and 2,388 validation headlines.

- The dataset's validation split is the **test set**, used once at the end.
- `train` is split 90/10 (stratified, seeded) into **fit** and **dev**. Only dev is used for choices
  (TF-IDF regularisation here; nothing else is tuned).
- **73 headlines that occur in both train and test (after normalisation) are removed from train.** Duplicated headlines
  would inflate every model's score.
- The classes are imbalanced: **66% of the test set is Neutral**, so accuracy alone is misleading and macro-F1 is reported
  next to it, with a majority-class row as the floor.
- Every LLM variant is scored the same way, from the model's next-token probabilities over the three label words in a single
  forward pass, so the comparison is between models, not between output parsers. The tokenizer check is a unit-tested
  precondition: `Bearish`, `Bullish`, `Neutral` start with distinct tokens in Qwen's vocabulary.
- Intervals are percentile bootstrap over test examples; model-vs-model differences use the **exact paired McNemar test**,
  Holm-adjusted across comparisons.

## Results (test set, n = 2,388)

<!-- results:start -->
| model | accuracy (95% CI) | macro-F1 (95% CI) | ms / example | peak GPU GB | train min | trainable params |
|---|---|---|---|---|---|---|
| majority class | 0.656 (0.637-0.675) | 0.264 (0.260-0.269) | 0.0 |  |  |  |
| rules | 0.723 (0.705-0.740) | 0.591 (0.565-0.616) | 0.0 |  |  |  |
| tfidf+logreg | 0.822 (0.808-0.837) | 0.751 (0.729-0.772) | 0.1 |  |  |  |
| zero-shot | 0.381 (0.363-0.401) | 0.304 (0.285-0.324) | 13.9 |  |  |  |
| few-shot (8) | 0.748 (0.729-0.764) | 0.728 (0.707-0.746) | 55.9 |  |  |  |
| QLoRA fine-tuned | 0.918 (0.906-0.928) | 0.895 (0.880-0.909) | 23.7 | 7.2 | 25 | 18.5M |

Setup: `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16, 2 epochs, lr 0.0002, seed 42, Tesla T4. fit 8523 / dev 947 / test 2388.

Paired exact McNemar versus TF-IDF + logistic regression (Holm-adjusted across comparisons):

| model | model-only correct | TF-IDF-only correct | p | p (Holm) |
|---|---|---|---|---|
| majority class | 129 | 527 | 5.9e-58 | 2.3e-57 |
| rules | 171 | 408 | 2.5e-23 | 5.0e-23 |
| zero-shot | 177 | 1232 | 1.0e-194 | 5.2e-194 |
| few-shot (8) | 280 | 458 | 5.9e-11 | 5.9e-11 |
| QLoRA fine-tuned | 297 | 70 | 2.1e-34 | 6.4e-34 |
<!-- results:end -->

### What the numbers say

- **Fine-tuning clearly beats the strongest baseline.** QLoRA reaches **0.918 accuracy / 0.895 macro-F1** against TF-IDF + logistic regression's 0.822 / 0.751 (+9.6 and +14.4 points; the
  intervals do not overlap). On the same 2,388 headlines the fine-tuned model is right where TF-IDF is wrong on 297 and the reverse on only 70 (exact McNemar, p of order 1e-34 after Holm).
- **The gain is concentrated in the minority classes.** Per-class F1, TF-IDF vs fine-tuned: Bearish 0.661 vs 0.863, Bullish 0.705 vs 0.882, Neutral 0.886 vs 0.940. A model that only
  learns "mostly Neutral" is capped near 0.66 accuracy; the fine-tuned one is not.
- **What it costs.** 25 minutes of training on a T4, 7.2 GB peak GPU memory, 18.5M trainable parameters (about 1.2% of the model), and **23.7 ms per example on the GPU against 0.1 ms
  for TF-IDF on one CPU core**: roughly 200x slower per example, and it needs a GPU. About ten accuracy points for two orders of magnitude of inference cost is a real trade, not a free win.
- **Few-shot prompting lands between the baselines** (0.748 / 0.728): better than the keyword rules on macro-F1, well short of TF-IDF, and the slowest at 55.9 ms because of the long prompt.
- **The zero-shot number (0.381, below always-Neutral) is a bias artifact, not a verdict on LLM sentiment ability.** The untuned model answers "Bullish" for 78% of headlines (Neutral 20%,
  Bearish 1.5%) under this prompt and next-token readout, and no calibration was applied. Reading it as "LLMs cannot do this" would be wrong; reading it as "an uncalibrated readout of a
  small base model is unusable here" is right.
- **Where the fine-tuned model still errs** (197 of 2,388): almost entirely Neutral vs directional confusions (Bullish->Neutral 57, Neutral->Bearish 46, Neutral->Bullish 44, Bearish->Neutral 40);
  outright Bearish/Bullish swaps are only 10. Several sampled errors are debatable labels (for example "Margins suffer at Ituran Location in Q3" is labelled Neutral and predicted Bearish),
  which suggests part of the remaining error is label noise, though this was not measured.

## Run it

1. Open `notebooks/fxnews_qlora.ipynb` in Google Colab, set the runtime to **T4 GPU**, *Run all* (about 35-50 minutes).
2. Download `fxnews_results.zip` (LoRA adapter, `results.json`, training curve).
3. `python scripts/fill_readme.py path/to/results.json` fills the table above and the comparison section.

The notebook embeds the library source from `src/fxnews` (generated by `build_notebook.py`), so the code that runs on Colab is exactly the
code under test. Local checks need no GPU:

```bash
pip install -e ".[dev]"
pytest                      # data/protocol, metrics, statistics, prompt and loss-masking construction, tokenizer preconditions
python build_notebook.py    # regenerate the notebook after editing src/
```

## Method details that matter

- **Loss is on the answer only.** Prompt tokens are masked to -100; the model learns the label word and the end-of-sequence
  token (a test checks this on a stand-in tokenizer, and the real Qwen tokenizer was checked by hand: the learned tokens are
  `Bear`, `ish`, `<|im_end|>`).
- **Plain `Trainer`, not TRL.** The training loop uses only `transformers` + `peft` + `bitsandbytes`, which have stable APIs; a
  wrapper library would add a version-sensitivity the experiment does not need.
- **4-bit NF4 with double quantisation, fp16 compute** (a T4 has no bf16), paged 8-bit AdamW, gradient checkpointing, cosine schedule.
- **Few-shot demonstrations** come from the *fit* split, class-balanced (3/3/2), fixed seed; never from dev or test.
- Training time, peak GPU memory and trainable-parameter count are recorded alongside accuracy, because "better" without the cost is half an answer.

## Limitations

- **One dataset, one seed, one run, one model size.** A single QLoRA run has seed-to-seed variance this design does not measure; the 9.6-point margin is large relative to the intervals, but it is one run.
- **Short, noisy tweets.** The labels come from the dataset's annotators; some label noise is inherent, which caps every model.
- **The rules baseline is hand-written** by the repo author before any model was run, with short word lists; a better lexicon would raise it.
- **No hyperparameter search for the LLM.** One configuration, chosen in advance (the optional ablation cell compares LoRA ranks on dev only).
- **Latency is measured on the Colab T4 at batch size 32 and TF-IDF on one CPU core**: not a like-for-like hardware comparison, only an order-of-magnitude one.

MIT licence (code). The dataset and the Qwen weights have their own licences.
