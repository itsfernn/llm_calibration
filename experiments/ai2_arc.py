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
import time

END_THINK_TOKEN_ID = 151668

ANSWER_PREFIX = """{
    \"answer\": \""""

DEFAULT_SYSTEM_PROMPT = """Answer the following multiple-choice question. Choose the correct answer from the choices provided.
Then output ONLY the following JSON (no extra text):

{
  "answer": <final numeric answer>,
  "confidence": <float between 0 and 1>
}

Rules:
- answer must be the label of the correct choice
- confidence must be a float between 0 and 1 (e.g. 0.82)
- no extra text"""


def format_question(q):
    lines = []

    lines.append(q.get("question", ""))
    lines.append("")

    choices = q.get("choices", {})
    texts = choices.get("text", [])
    labels = choices.get("label", [])

    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")

    return "\n".join(lines)


def run_ai2_arc(
    model,
    dataset_config="ARC-Challenge",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    batch_size=4,
    out_dir="out_runs",
    max_samples=None,
    thinking=False,
    **kwargs,
):
    """
    Run AI2 ARC evaluation on a model.

    Outputs are written to `run_dir/outputs.jsonl` (JSON lines format).
    Returns the path to the run directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Setup output dir
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Metadata
    metadata = {
        "model": model,
        "dataset": "ai2_arc",
        "dataset_config": dataset_config,
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
    test_data = load_dataset("ai2_arc", dataset_config, split="test")

    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model,
        dtype="auto",
        attn_implementation="sdpa",
        device_map="auto",
    )

    model.eval()
    print(f"Model dtype: {model.dtype}")
    metadata["dtype"] = str(model.dtype)
    metadata["attention"] = "sdpa"

    if device.type == "cuda":
        print(
            f"GPU memory used after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    answer_prefix_len = len(tokenizer(ANSWER_PREFIX)["input_ids"])

    gen_kwargs = dict(do_sample=False)

    output_path = os.path.join(run_dir, "outputs.jsonl")
    f_out = open(output_path, "w")

    # --- PERFORMANCE FIX: Pre-format and Pre-tokenize Dataset via Batched Map ---
    print("Pre-processing dataset chat templates and token IDs...")
    t_prep = time.perf_counter()

    def preprocess_batch(examples):
        base_texts = []
        keys = list(examples.keys())
        num_items = len(examples[keys[0]])

        for i in range(num_items):
            row = {k: examples[k][i] for k in keys}
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_question(row)},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            base_texts.append(text)

        out_dict = {"base_text": base_texts}

        if not thinking:
            # If thinking is disabled, we can pre-compile the entire execution prompt right here
            full_texts = [t + ANSWER_PREFIX for t in base_texts]
            tokenized = tokenizer(full_texts, padding=False, truncation=True)
            out_dict["input_ids"] = tokenized["input_ids"]
            out_dict["attention_mask"] = tokenized["attention_mask"]
        else:
            # If thinking is enabled, we pre-compile stage 1 (before thinking occurs)
            tokenized = tokenizer(base_texts, padding=False, truncation=True)
            out_dict["input_ids"] = tokenized["input_ids"]
            out_dict["attention_mask"] = tokenized["attention_mask"]

        return out_dict

    test_data = test_data.map(preprocess_batch, batched=True, desc="Mapping dataset")
    print(f"Dataset pre-processed in {time.perf_counter() - t_prep:.1f}s")
    # ----------------------------------------------------------------------------

    for start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
        end = min(start + batch_size, len(test_data))
        batch = [test_data[i] for i in range(start, end)]

        if thinking:
            # Fast Tensor Dynamic Padding using pre-tokenized base IDs
            batch_features = [
                {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                for b in batch
            ]
            inputs = tokenizer.pad(
                batch_features, padding=True, return_tensors="pt"
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
                b["base_text"] + thinking_texts[i] + "\n\n" + ANSWER_PREFIX
                for i, b in enumerate(batch)
            ]

            # Re-tokenization is required here as text length depends on dynamic thinking generation
            new_inputs = tokenizer(
                new_texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

        else:
            # OPTIMIZATION AT WORK: Zero string manipulation, zero text tokenization in this path.
            # We directly pass the pre-compiled full token sequences into the rapid padding utility.
            batch_features = [
                {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                for b in batch
            ]
            new_inputs = tokenizer.pad(
                batch_features, padding=True, return_tensors="pt"
            ).to(device)
            thinking_texts = None

        with torch.inference_mode():
            out = model.generate(
                **new_inputs,
                max_new_tokens=100,
                output_scores=True,
                return_dict_in_generate=True,
                **gen_kwargs,
            )
            probs = F.softmax(out.scores[0], dim=-1)
            label_probs = []
            for i, b in enumerate(batch):
                labels = b["choices"]["label"]
                label_ids = tokenizer.convert_tokens_to_ids(labels)
                label_probs.append(probs[i, label_ids].tolist())

        content_sequences = out.sequences[
            :, new_inputs["input_ids"].shape[1] - answer_prefix_len :
        ]

        for i, b in enumerate(batch):
            content_ids = content_sequences[i].tolist()
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip(
                "\n"
            )

            # Extract answer
            parsed = parse_output(content)
            prediction = parsed["answer"]
            confidence = parsed["confidence"]

            output = {
                "id": b["id"],
                "question": b["question"],
                "answer": b["answerKey"],
                "labels": b["choices"]["label"],
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
