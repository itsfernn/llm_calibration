import os
import json
import datetime
import subprocess
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from utils import parse_output, find_subsequence, print_metadata

END_THINK_TOKEN_ID = 151668

ANSWER_PREFIX = """{
    \"answer\": \""""

DEFAULT_SYSTEM_PROMPT = """Answer the following multiple-choice question. Choose the correct answer from the choices provided.
Then output ONLY the following JSON (no extra text):

{
  "answer": <answer label>,
  "confidence": <float between 0 and 1>
}

Rules:
- answer must be the label of the correct choice
- confidence must be a float between 0 and 1 (e.g. 0.82)
- no extra text"""

MMLU_LABELS = ["A", "B", "C", "D"]


def format_question(q):
    lines = []

    lines.append(q.get("question", ""))
    lines.append("")

    choices = q.get("choices", [])
    for label, text in zip(MMLU_LABELS, choices):
        lines.append(f"{label}. {text}")

    return "\n".join(lines)


def run_mmlu(
    model,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    batch_size=4,
    out_dir="out_runs",
    max_samples=None,
    thinking=False,
    **kwargs,
):
    """
    Run MMLU evaluation on a model.

    Outputs are written to `run_dir/outputs.jsonl` (JSON lines format).
    Returns the path to the run directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup output dir
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Metadata
    metadata = {
        "model": model,
        "dataset": "cais/mmlu",
        "batch_size": batch_size,
        "thinking": thinking,
        "timestamp": run_timestamp,
        "device": str(device),
        "max_samples": max_samples,
        "system_prompt": system_prompt,
    }

    try:
        metadata["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        metadata["git_commit"] = "unknown"

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print_metadata(metadata)

    # Load dataset
    test_data = load_dataset("cais/mmlu", "all", split="test")

    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, dtype="auto", device_map="auto")
    model.eval()

    answer_prefix_len = len(tokenizer(ANSWER_PREFIX)["input_ids"])

    gen_kwargs = dict(do_sample=False)

    output_path = os.path.join(run_dir, "outputs.jsonl")
    f_out = open(output_path, "w")

    tokenizer.padding_side = "left"

    for start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
        end = min(start + batch_size, len(test_data))
        batch = [test_data[i] for i in range(start, end)]

        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_question(row)},
            ]
            for row in batch
        ]

        texts = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

        thinking_texts = None
        if thinking:
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    eos_token_id=END_THINK_TOKEN_ID,
                    max_new_tokens=1000,
                    **gen_kwargs,
                )

            thinking_sequences = out[:, inputs["input_ids"].shape[1] :]
            thinking_texts = tokenizer.batch_decode(
                thinking_sequences, skip_special_tokens=True
            )

            new_texts = [
                t + thinking_texts[i] + "\n\n" + ANSWER_PREFIX
                for i, t in enumerate(texts)
            ]

        else:
            new_texts = [t + ANSWER_PREFIX for t in texts]

        new_inputs = tokenizer(
            new_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        label_ids = tokenizer.convert_tokens_to_ids(MMLU_LABELS)
        with torch.inference_mode():
            logits = model(**new_inputs).logits
            label_probs = []
            for i in range(len(batch)):
                label_probs.append(logits[i, -1, label_ids].softmax(dim=-1).tolist())

        with torch.inference_mode():
            out = model.generate(
                **new_inputs,
                max_new_tokens=100,
                **gen_kwargs,
            )

        content_sequences = out[
            :, new_inputs["input_ids"].shape[1] - answer_prefix_len :
        ]

        for i, b in enumerate(batch):
            content_ids = content_sequences[i].tolist()
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip(
                "\n"
            )

            # Extract answer
            output = parse_output(content)
            prediction = output["answer"]
            confidence = output["confidence"]

            answer_label = MMLU_LABELS[b["answer"]]

            output = {
                "id": f"{b['subject']}_{start + i}",
                "subject": b["subject"],
                "question": b["question"],
                "answer": answer_label,
                "labels": MMLU_LABELS,
                "content": content,
                "thinking": thinking_texts[i] if thinking else None,
                "prediction": prediction,
                "label_probs": label_probs[i],
                "verb_conf": confidence,
            }
            json.dump(output, f_out)
            f_out.write("\n")

    f_out.close()

    print(f"Done. Saved to {run_dir}")
    return run_dir
