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
  "answer": <final numeric answer>,
  "confidence": <float between 0 and 1>
}

Rules:
- answer must be the label of the correct choice
- confidence must be a float between 0 and 1 (e.g. 0.82)
- no extra text"""

def format_question(q):
    lines = []

    lines.append(q.get('question', ''))
    lines.append("")

    choices = q.get('choices', {})
    texts = choices.get('text', [])
    labels = choices.get('label', [])

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
    **kwargs
):
    """
    Run AI2 ARC evaluation on a model.
    
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
        "dataset": "ai2_arc", 
        "dataset_config": dataset_config,
        "batch_size": batch_size,
        "thinking": thinking,
        "timestamp": run_timestamp,
        "device": str(device),
        "max_samples": max_samples,
        "system_prompt": system_prompt
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

    # Load dataset
    test_data = load_dataset("ai2_arc", dataset_config, split="test")

    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        dtype="auto",
        device_map="auto"
    )
    model.eval()


    answer_prefix_len = len(tokenizer(ANSWER_PREFIX)["input_ids"])


    ## Hyperparameters from the HuggingFace model card for qwen
    if thinking:
        gen_kwargs = dict(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    else:
        gen_kwargs = dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

    output_path = os.path.join(run_dir, "outputs.jsonl")
    f_out = open(output_path, "w")

    tokenizer.padding_side = "left"


    for start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
        end = min(start + batch_size, len(test_data))
        batch = test_data.select(range(start, end))

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


        if thinking:
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    eos_token_id=END_THINK_TOKEN_ID,
                    max_new_tokens=1000,
                    return_dict_in_generate=True,
                    **gen_kwargs,
                )

            thinking_sequences = out.sequences[:, inputs["input_ids"].shape[1]:]
            thinking_texts = tokenizer.batch_decode(thinking_sequences, skip_special_tokens=True)

            new_texts = [ t + thinking_texts[i] + "\n\n" + ANSWER_PREFIX for i, t in enumerate(texts) ]

        else:
            new_texts = [ t + ANSWER_PREFIX  for t in texts ]

        new_inputs = tokenizer(
            new_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.inference_mode():
             logits = model(**new_inputs).logits
             label_probs = []
             for i, b in enumerate(batch):
                 labels = b["choices"]["label"]
                 label_ids = tokenizer.convert_tokens_to_ids(labels)
                 label_probs.append(logits[i, -1, label_ids].softmax(dim=-1).tolist())


        with torch.inference_mode():
            out = model.generate(
                **new_inputs,
                max_new_tokens=100,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

        sequences = out.sequences
        content_sequences = sequences[:, new_inputs["input_ids"].shape[1] - answer_prefix_len:]

        for i, b in enumerate(batch):
            content_ids = content_sequences[i].tolist()
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n")


            # Extract answer
            output = parse_output(content)
            prediction = output["answer"]
            confidence = output["confidence"]

            output = {
                "id": b["id"],
                "question": b["question"],
                "answer":  b["answerKey"],
                "labels": b["choices"]["label"],
                "content": content,
                "thinking": thinking_texts[i] if thinking else None,
                "prediction": prediction,
                "label_probs": label_probs[i],
                "verb_conf": confidence
            }
            json.dump(output, f_out)
            f_out.write("\n")
            
        # Cleanup per batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    f_out.close()

    print(f"Done. Saved to {run_dir}")
    return run_dir
