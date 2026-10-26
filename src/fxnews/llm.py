"""Prompting, label scoring and QLoRA training for a small causal LM.

All three LLM variants (zero-shot, few-shot, fine-tuned) are scored the same way: the model's next-token
probabilities over the three label words, read from ONE forward pass. This avoids free-text parsing (no
"the answer seems to be bullish-ish" ambiguity) and makes the comparison between prompting and fine-tuning
a comparison of the models, not of how forgiving the output parser is.

The torch/transformers/peft imports are inside functions so the pure parts (prompts, example construction,
collation) can be unit-tested on a machine without a GPU or those libraries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fxnews.data import LABELS

INSTRUCTION = "Classify the sentiment of this financial news headline for the asset it mentions as Bearish, Bullish or Neutral."
MAX_LEN = 128


def build_prompt(text: str, shots: Sequence[tuple[str, int]] = ()) -> str:
    """Plain-text prompt (no chat template, so it behaves the same across model families). The answer follows 'Sentiment:'."""
    parts = [INSTRUCTION, ""]
    for t, y in shots:
        parts += [f"Headline: {t}", f"Sentiment: {LABELS[y]}", ""]
    parts += [f"Headline: {text}", "Sentiment:"]
    return "\n".join(parts)


def choose_shots(texts: Sequence[str], labels: np.ndarray, per_class: Sequence[int] = (3, 3, 2), seed: int = 0) -> list[tuple[str, int]]:
    """A fixed, class-balanced set of demonstrations drawn from the FIT split (never from DEV or test)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    shots = []
    for c, k in enumerate(per_class):
        idx = np.flatnonzero(labels == c)
        shots += [(texts[i], c) for i in rng.choice(idx, k, replace=False)]
    order = rng.permutation(len(shots))  # interleave classes so the last demonstration is not always the same label
    return [shots[i] for i in order]


def make_sft_example(tokenizer, text: str, label: int, max_len: int = MAX_LEN) -> dict[str, list[int]]:
    """Prompt tokens are masked out of the loss (-100); only the label word and the end-of-sequence token are learned."""
    prompt_ids = tokenizer(build_prompt(text), add_special_tokens=False)["input_ids"][: max_len - 8]
    target_ids = tokenizer(" " + LABELS[label], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad to the longest sequence in the batch."""
    n = max(len(b["input_ids"]) for b in batch)
    return {
        "input_ids": [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch],
        "attention_mask": [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch],
        "labels": [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch],
    }


def label_token_ids(tokenizer) -> list[int]:
    """First token of each ' <Label>'. They must differ, otherwise one forward pass cannot tell the labels apart."""
    ids = [tokenizer(" " + lab, add_special_tokens=False)["input_ids"][0] for lab in LABELS]
    if len(set(ids)) != len(ids):
        raise ValueError(f"label words share a first token ({ids}); pick different label words for this tokenizer")
    return ids


def load_model(name: str, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # so the last position of every row is the position that predicts the answer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.float16}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(name, **kwargs), tok


def score(model, tok, texts: Sequence[str], shots: Sequence[tuple[str, int]] = (), batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (probabilities [n, 3], predictions [n], milliseconds per example)."""
    import time

    import torch

    ids = torch.tensor(label_token_ids(tok), device=model.device)
    model.eval()
    probs = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok([build_prompt(t, shots) for t in texts[i : i + batch_size]], return_tensors="pt", padding=True, truncation=True,
                      max_length=1024, add_special_tokens=False).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            probs.append(torch.softmax(logits[:, ids].float(), dim=-1).cpu().numpy())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / len(texts)
    p = np.concatenate(probs)
    return p, p.argmax(1), ms


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, cfg)


def train_lora(model, tok, fit_x, fit_y, dev_x, dev_y, out_dir: str, epochs: int = 2, lr: float = 2e-4, batch_size: int = 16, grad_accum: int = 2,
               seed: int = 42):
    """Plain `Trainer` (no TRL), so the training loop does not depend on a fast-moving wrapper API. Returns the trainer."""
    import torch
    from transformers import Trainer, TrainingArguments

    class DS(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.items = [make_sft_example(tok, x, int(y)) for x, y in zip(xs, ys, strict=True)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collator(batch):
        return {k: torch.tensor(v) for k, v in collate(batch, tok.pad_token_id).items()}

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0, fp16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True, logging_steps=25, eval_strategy="steps", eval_steps=200, save_strategy="no",
        report_to="none", seed=seed, remove_unused_columns=False,
    )
    model.config.use_cache = False
    trainer = Trainer(model=model, args=args, train_dataset=DS(fit_x, fit_y), eval_dataset=DS(dev_x, dev_y), data_collator=collator)
    trainer.train()
    model.config.use_cache = True
    return trainer
