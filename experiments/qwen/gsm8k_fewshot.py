"""GSM8K few-shot evaluation for the post-training (base vs. aligned) comparison.

Why this script exists
----------------------
The white-box results (Section 5.1 of the thesis) show that verbalized confidence
collapses to values near 1.0 across all Qwen3 sizes. One candidate explanation is
that preference-based post-training (RLHF/RLVR) biases models toward confident
responses. To test it, we compare a post-trained checkpoint (e.g. Qwen/Qwen3-8B)
against its non-RLHF base counterpart (e.g. Qwen/Qwen3-8B-Base) under an
*identical* elicitation protocol.

Design choices
--------------
- Plain-text format instead of JSON. Base models do not reliably follow the
  chat template or complex JSON instructions, so we use the minimal line format

      Answer: <number>
      Confidence: <float between 0 and 1>

  The *same* format is used for both checkpoints, so the only difference between
  the two runs is post-training.
- No chat template. Prompts are assembled as plain text for both checkpoints
  (base models do not recognize the chat template).
- Few-shot exemplars demonstrate the format. Exemplar confidence values are
  deliberately varied (1.0, 0.9, 0.85, 0.7) so the model copies the *format* and
  not an anchor at 1.0.
- Thinking mode is disabled for both checkpoints (the main size grid is also
  non-thinking; thinking would add a confound).
- The output schema (outputs.jsonl columns) matches the main gsm8k.py runner, so
  the existing notebook preprocessing (src/utils, notebooks/qwen.py) works
  unchanged: `logprobs` holds per-token log-probs of the answer span,
  `verb_conf` the verbalized confidence, `prediction` the parsed answer string.

Usage
-----
    cd experiments/qwen
    python run.py --dataset gsm8k_fewshot --model Qwen/Qwen3-8B --no-thinking \\
        --batch_size 32 --n_examples 3 --tags fewshot_comp
    python run.py --dataset gsm8k_fewshot --model Qwen/Qwen3-8B-Base --no-thinking \\
        --batch_size 32 --n_examples 3 --tags fewshot_comp
"""

import datetime
import json
import os
import re
import subprocess

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import score_sequences
from utils import print_metadata

ANSWER_PREFIX = "\nAnswer:"

# Short arithmetic exemplars with deliberately varied confidence values.
DEFAULT_EXAMPLES = [
    ("What is 7 times 8?", "56", 1.0),
    ("A train travels at 60 miles per hour. How far does it go in 3 hours?", "180", 0.9),
    ("Sarah has 5 apples and gives 2 away. How many are left?", "3", 0.85),
    ("A shirt costs $20 and is 25% off. What is the sale price?", "15", 0.7),
]

INSTRUCTION = """Answer the math question and estimate your confidence in your answer.

Format:
Answer: <number>
Confidence: <float between 0 and 1, where 0.0 = pure guess and 1.0 = absolutely certain>

Examples:
{examples}

Now answer this question:"""


def build_fewshot_prompt(question, examples, n_examples):
    """Assemble the plain-text few-shot prompt for one question."""
    example_text = "\n\n".join(
        f"Question: {q}\nAnswer: {a}\nConfidence: {c}"
        for q, a, c in examples[:n_examples]
    )
    return INSTRUCTION.format(examples=example_text) + f"\n\nQuestion: {question}"


def parse_simple_output(text):
    """Tolerant parser for the Answer:/Confidence: line format.

    The prompt already ends with the "Answer:" label, so the model's
    continuation typically starts directly with the number (e.g. " 180\\n
    Confidence: 0.9"). We therefore prefer a number after an explicit
    "Answer:" label (in case the model repeats it) and otherwise take the
    first number before the confidence line. Confidence is clamped to [0, 1];
    a missing answer or confidence yields None (the notebook drops rows with
    missing verb_conf via notna()).
    """
    result = {"answer": None, "confidence": None}
    if not text or not isinstance(text, str):
        return result

    # answer: number after an explicit "Answer:" label, if present
    m = re.search(r"[Aa]nswer\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        result["answer"] = m.group(1)
    else:
        # fallback: first number before the confidence line (continuation)
        conf_pos = text.find("Confidence")
        region = text[:conf_pos] if conf_pos != -1 else text
        m = re.search(r"-?\d+(?:\.\d+)?", region)
        if m:
            result["answer"] = m.group(0)

    m = re.search(r"[Cc]onfidence\s*:\s*([+-]?[0-9]*\.?[0-9]+)", text)
    if m:
        result["confidence"] = max(0.0, min(1.0, float(m.group(1))))

    return result


def run_gsm8k_fewshot(
    model,
    batch_size,
    out_dir="out_runs",
    max_samples=None,
    n_examples=3,
    examples=None,
    tags=None,
    thinking=False,
    debug=False,
    **kwargs,
):
    """Run the GSM8K few-shot post-training comparison on one checkpoint.

    Outputs are written to `run_dir/outputs.jsonl`; the run directory path is
    returned. The `metadata.json` records the few-shot protocol (n_examples,
    plain-text format, exemplars) so the comparison is reproducible.
    """
    if thinking:
        print("Warning: thinking mode is ignored for the few-shot comparison; "
              "both checkpoints run non-thinking.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    if examples is None:
        examples = DEFAULT_EXAMPLES
    if tags is None:
        tags = ["fewshot_comp"]

    metadata = {
        "model": model,
        "dataset": "gsm8k",
        "batch_size": batch_size,
        "thinking": False,
        "system_prompt": INSTRUCTION,
        "prompt_format": "plain",
        "few_shot": True,
        "n_examples": n_examples,
        "examples": [
            {"question": q, "answer": a, "confidence": c} for q, a, c in examples[:n_examples]
        ],
        "tags": tags,
        "type": "float",
        "timestamp": run_timestamp,
        "device": str(device),
        "max_samples": max_samples,
    }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception:
        metadata["git_commit"] = "unknown"

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print_metadata(metadata)

    # Dataset
    test_data = load_dataset("gsm8k", "main", split="test")
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model, dtype="auto", device_map="auto")
    model.eval()

    answer_prefix_ids = tokenizer.encode(
        "\n" + ANSWER_PREFIX,
        add_special_tokens=False,
    )

    # Non-thinking decoding settings (same as the main non-thinking grid runs)
    gen_kwargs = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

    def preprocess_batch(examples_batch):
        base_texts = [
            build_fewshot_prompt(q, examples, n_examples)
            for q in examples_batch["question"]
        ]
        # Force the model to continue from the Answer: line
        full_texts = [t + ANSWER_PREFIX for t in base_texts]
        tokenized = tokenizer(full_texts, padding=False, truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    test_data = test_data.map(preprocess_batch, batched=True, batch_size=batch_size)

    output_path = os.path.join(run_dir, "outputs.jsonl")
    f_out = open(output_path, "w")

    for start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
        end = min(start + batch_size, len(test_data))
        batch = test_data.select(range(start, end))

        batch_features = [
            {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
            for b in batch
        ]
        final_inputs = tokenizer.pad(
            batch_features, return_tensors="pt", padding=True
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **final_inputs,
                max_new_tokens=100,
                **gen_kwargs,
            )

        input_len = final_inputs["input_ids"].shape[1] - len(answer_prefix_ids)
        content_sequences = out[:, input_len:]
        content_texts = tokenizer.batch_decode(
            content_sequences, skip_special_tokens=True
        )

        out_features = {
            "input_ids": out,
            "attention_mask": (out != tokenizer.pad_token_id).long(),
        }
        log_probs = score_sequences(model, out_features)

        outputs = []
        for i, b in enumerate(batch):
            content = content_texts[i]
            content_ids = content_sequences[i].tolist()

            parsed = parse_simple_output(content)
            prediction = parsed["answer"]
            confidence = parsed["confidence"]

            prediction_logprobs = []
            if prediction is not None:
                start_idx = content.find(prediction)
                if start_idx != -1:
                    end_idx = start_idx + len(prediction)
                    current_len = 0
                    for step_idx, token_id in enumerate(content_ids):
                        if current_len >= end_idx:
                            break
                        extended_text = tokenizer.decode(
                            content_ids[: step_idx + 1], skip_special_tokens=True
                        )
                        next_len = len(extended_text)
                        if next_len > start_idx and current_len < end_idx:
                            global_token_idx = input_len + step_idx
                            if global_token_idx > 0:
                                prediction_logprobs.append(
                                    log_probs[i, global_token_idx - 1].item()
                                )
                        current_len = next_len
                else:
                    print(
                        f"Warning: Prediction '{prediction}' not found in decoded text."
                    )

            output = {
                "index": start + i,
                "question": b["question"],
                "answer": b["answer"],
                "content": content,
                "thinking": None,
                "n_thinking_tokens": None,
                "prediction": prediction,
                "logprobs": prediction_logprobs,
                "verb_conf": confidence,
            }
            outputs.append(output)

        if debug:
            for o in outputs:
                print(f"--- Sample {o['index']} ---")
                print(f"Question: {o['question']}")
                print(f"Generated: {o['content']!r}")
                print(f"Prediction: {o['prediction']} (confidence: {o['verb_conf']})")
                if o["logprobs"]:
                    avg_lp = sum(o["logprobs"]) / len(o["logprobs"])
                    print(f"Logprobs: avg={avg_lp:.4f}, tokens={len(o['logprobs'])}")
                else:
                    print("Logprobs: none")

        for o in outputs:
            json.dump(o, f_out)
            f_out.write("\n")

    f_out.close()
    print(f"Done. Saved to {run_dir}")
    return run_dir
