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

DEFAULT_SYSTEM_PROMPT = """Solve the following math problem.

Please reason step by step, and put your final answer within \boxed{}.

Then output ONLY the following JSON (no extra text):

{
  "answer": <final numeric answer>,
  "confidence": <float between 0 and 1>
}

Rules:
- The answer must exactly match the value inside \boxed{}.
- Confidence must be a decimal between 0 and 1 (e.g., 0.82).
- Do not include units in the answer.
- Do not include any explanation inside or after the JSON."""

def score_sequences(model, sequences):
    """
    Compute logprobs for actual tokens in sequences (prompt + generation)
    Returns token_logprobs tensor of shape (batch, seq_len-1)
    """
    with torch.no_grad():
        outputs = model(input_ids=sequences)

    logits = outputs.logits[:, :-1, :]
    targets = sequences[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    return token_logprobs

def run_gsm8k(
    model_name,
    system_prompt,
    batch_size=4,
    out_dir="out_runs",
    max_samples=None,
    thinking=True,
):
    """
    Run GSM8K evaluation on a model.
    
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
        "model": model_name,
        "batch_size": batch_size,
        "system_prompt": system_prompt,
        "timestamp": run_timestamp,
        "device": str(device),
        "max_samples": max_samples
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
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

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

    outputs_log = []

    for start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
        batch = test_data[start:start + batch_size]
        questions = batch["question"]
        answers = batch["answer"]

        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ]
            for q in questions
        ]

        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for messages in messages_batch
        ]

        tokenizer.padding_side = "left"

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )

        sequences = out.sequences

        # [steps, batch, vocab]
        score_tensor = torch.stack(out.scores, dim=0)
        log_probs = F.log_softmax(score_tensor, dim=-1)

        input_len = inputs["input_ids"].shape[1]
        output_sequences = sequences[:, input_len:]

        for i in range(len(questions)):
            # parsing thinking content
            output_ids = output_sequences[i].tolist()
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_ids = output_ids[:index]
            content_ids = output_ids[index:]

            thinking_content = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n")


            # Extract answer
            output = parse_output(content)
            prediction = output["answer"]
            confidence = output["confidence"]

            ### Deprecated: using regex parsing instead of JSON parsing for better robustness
            #prediction = extract_key(content, "answer")
            #confidence = extract_key(content, "confidence")

            prediction_logprobs = []

            if prediction is not None:
                prediction_ids = tokenizer(
                    prediction,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"][0].to(device)

                start_idx = find_subsequence(content_ids, prediction_ids)

                if start_idx is not None:
                    for t, token_id in enumerate(prediction_ids):
                        step_idx = start_idx + t

                        if step_idx >= log_probs.shape[0]:
                            break

                        lp = log_probs[step_idx, i, token_id]
                        prediction_logprobs.append(lp.item())

            outputs_log.append({
                "index": start + i,
                "question": questions[i],
                "answer": answers[i],
                "content": content,
                "thinking": thinking_content,
                "prediction": prediction,
                "logprobs": prediction_logprobs,
                "verb_conf": confidence
            })

    with open(os.path.join(run_dir, "outputs.json"), "w") as f:
        json.dump(outputs_log, f, indent=2)

    print(f"Done. Saved to {run_dir}")
    return run_dir
