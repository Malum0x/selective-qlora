"""
Evaluate base, baseline-finetuned, and filtered-finetuned models on
out-of-distribution reasoning benchmarks.

Metrics:
  - ARC-Challenge: multiple-choice accuracy (A/B/C/D)
  - GSM8K: math word problem accuracy (exact number match)
  - Sample generations for qualitative comparison

Run from: /home/bart/Desktop/selective-qlora/selective-qlora/
    python eval_all.py
"""

import re
import gc
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL_PATH   = "base_model"
BASELINE_ADAPTER  = "baseline_results/final_adapter"
FILTERED_ADAPTER  = "filtered_results/final_adapter"

ARC_SAMPLES  = 200   # ARC-Challenge test has 1172; take first 200 for speed
GSM8K_SAMPLES = 100  # GSM8K test has 1319; take first 100

QUAL_PROMPTS = [
    "If a train travels at 60 mph and needs to cover 210 miles, how long will the journey take? Show your reasoning.",
    "What is the difference between mitosis and meiosis?",
    "Write a Python function that returns the nth Fibonacci number using dynamic programming.",
    "A store sells apples for $0.50 each and oranges for $0.75 each. If someone buys 4 apples and 3 oranges, how much do they spend?",
]
# ─────────────────────────────────────────────────────────────────────────────


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config(), device_map="auto"
    )
    return model, tok


def load_with_adapter(base_path, adapter_path):
    model, tok = load_base(base_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tok


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Generation ───────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=256, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── ARC-Challenge ─────────────────────────────────────────────────────────────

def build_arc_prompt(example):
    """Format an ARC question as a multiple-choice prompt."""
    choices = example["choices"]
    letters = choices["label"]   # e.g. ["A","B","C","D"]
    texts   = choices["text"]
    options = "\n".join(f"{l}. {t}" for l, t in zip(letters, texts))
    return (
        f"Question: {example['question']}\n"
        f"{options}\n"
        f"Answer with only the letter (A, B, C, or D):"
    )


def eval_arc(model, tokenizer, samples):
    correct = 0
    for ex in tqdm(samples, desc="  ARC", leave=False):
        prompt = build_arc_prompt(ex)
        response = generate(model, tokenizer, prompt, max_new_tokens=8)
        # Extract the first capital letter A-D from the response
        match = re.search(r"\b([A-D])\b", response.upper())
        predicted = match.group(1) if match else ""
        if predicted == ex["answerKey"].upper():
            correct += 1
    return correct / len(samples)


# ── GSM8K ────────────────────────────────────────────────────────────────────

SYSTEM_GSM8K = (
    "Solve the math problem step by step. "
    "At the end of your answer, write the final number on its own line prefixed with '####'."
)


def extract_gsm8k_answer(text):
    """Extract the number after #### in the model output."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in response
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def eval_gsm8k(model, tokenizer, samples):
    correct = 0
    for ex in tqdm(samples, desc="  GSM8K", leave=False):
        response = generate(
            model, tokenizer, ex["question"],
            max_new_tokens=300, system=SYSTEM_GSM8K
        )
        predicted = extract_gsm8k_answer(response)
        gold = extract_gsm8k_answer(ex["answer"])
        if predicted == gold:
            correct += 1
    return correct / len(samples)


# ── Qualitative ───────────────────────────────────────────────────────────────

def eval_qualitative(model, tokenizer):
    print("\n  Sample generations:")
    for prompt in QUAL_PROMPTS:
        response = generate(model, tokenizer, prompt, max_new_tokens=256)
        snippet = response[:500] + ("…" if len(response) > 500 else "")
        print(f"\n  Q: {prompt}")
        print(f"  A: {snippet}")


# ── Main ─────────────────────────────────────────────────────────────────────

def load_benchmarks():
    print("Loading ARC-Challenge …")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_samples = list(arc_ds.select(range(min(ARC_SAMPLES, len(arc_ds)))))

    print("Loading GSM8K …")
    gsm_ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm_samples = list(gsm_ds.select(range(min(GSM8K_SAMPLES, len(gsm_ds)))))

    return arc_samples, gsm_samples


def evaluate(label, model, tokenizer, arc_samples, gsm_samples):
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)

    print("  Evaluating on ARC-Challenge …")
    arc_acc = eval_arc(model, tokenizer, arc_samples)
    print(f"  ARC-Challenge accuracy : {arc_acc:.1%}  ({int(arc_acc*len(arc_samples))}/{len(arc_samples)})")

    print("  Evaluating on GSM8K …")
    gsm_acc = eval_gsm8k(model, tokenizer, gsm_samples)
    print(f"  GSM8K accuracy         : {gsm_acc:.1%}  ({int(gsm_acc*len(gsm_samples))}/{len(gsm_samples)})")

    eval_qualitative(model, tokenizer)

    return {"arc_accuracy": arc_acc, "gsm8k_accuracy": gsm_acc}


def main():
    arc_samples, gsm_samples = load_benchmarks()

    results = {}

    # 1. Base model
    print("\n[1/3] Loading base model …")
    model, tok = load_base(BASE_MODEL_PATH)
    results["base"] = evaluate(
        "Base Model (Qwen2.5-3B-Instruct)", model, tok, arc_samples, gsm_samples
    )
    free(model)

    # 2. Baseline finetuned
    print("\n[2/3] Loading baseline finetuned model …")
    model, tok = load_with_adapter(BASE_MODEL_PATH, BASELINE_ADAPTER)
    results["baseline"] = evaluate(
        "Baseline Finetuned (random 10k from OpenHermes)", model, tok, arc_samples, gsm_samples
    )
    free(model)

    # 3. Filtered finetuned (selective QLoRA)
    print("\n[3/3] Loading filtered finetuned model …")
    model, tok = load_with_adapter(BASE_MODEL_PATH, FILTERED_ADAPTER)
    results["filtered"] = evaluate(
        "Filtered Finetuned (top 30% hardest — selective QLoRA)", model, tok, arc_samples, gsm_samples
    )
    free(model)

    # Summary
    bar = "=" * 65
    print(f"\n{bar}")
    print("  SUMMARY")
    print(bar)
    print(f"  {'Model':<44} {'ARC':>6} {'GSM8K':>7}")
    print(f"  {'-'*60}")
    labels = {
        "base":     "Base Model",
        "baseline": "Baseline Finetuned",
        "filtered": "Filtered Finetuned (selective QLoRA)",
    }
    for key, label in labels.items():
        r = results[key]
        print(f"  {label:<44} {r['arc_accuracy']:>6.1%} {r['gsm8k_accuracy']:>7.1%}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to eval_results.json")


if __name__ == "__main__":
    main()
