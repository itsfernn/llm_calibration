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
  "answer": """

DEFAULT_SYSTEM_PROMPT = """Solve the following math problem.

Please reason step by step, and put your final answer within  \\boxed{}.

Then output ONLY the following JSON (no extra text):

{
  "answer": <final numeric answer>,
  "confidence": <float between 0 and 1>
}

Rules:
- The answer must exactly match the value inside \\boxed{}.
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
    model,
    batch_size,
    thinking,
    out_dir="out_runs",
    max_samples=None,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    **kwargs
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
        "model": model,
        "batch_size": batch_size,
        "thinking": thinking,
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
    test_data = load_dataset("gsm8k", "main", split="test")

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
                {"role": "user", "content": q },
            ]
            for q in batch["question"]
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
            out = model.generate(
                **new_inputs,
                max_new_tokens=1000,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

        sequences = out.sequences
        content_sequences = sequences[:, new_inputs["input_ids"].shape[1] - answer_prefix_len:]
        output_log_probs = score_sequences(model, sequences)[:, new_inputs["input_ids"].shape[1] - answer_prefix_len:]


        for i, b in enumerate(batch):
            content_ids = content_sequences[i].tolist()
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip("\n")


            # Extract answer
            output = parse_output(content)
            prediction = output["answer"]
            confidence = output["confidence"]

            prediction_logprobs = []

            if prediction is not None:
                prediction_ids = tokenizer(prediction)["input_ids"]

                start_idx = find_subsequence(content_ids, prediction_ids)

                if start_idx is not None:
                    for t, token_id in enumerate(prediction_ids):
                        step_idx = start_idx + t
                        
                        # Verify token matches
                        if content_ids[step_idx] != token_id:
                            # Token mismatch, skip this prediction
                            print(f"Warning: Token mismatch at position {step_idx}: expected {token_id}, got {content_ids[step_idx]}")
                            prediction_logprobs = []
                            break
                        
                        lp = output_log_probs[i, step_idx]
                        prediction_logprobs.append(lp.item())

            output = {
                "index": start + i,
                "question": b["question"],
                "answer": b["answer"],
                "content": content,
                "thinking": thinking_texts[i] if thinking else None,
                "prediction": prediction,
                "logprobs": prediction_logprobs,
                "verb_conf": confidence
            }
            json.dump(output, f_out)
            f_out.write("\n")
    
    f_out.close()

    print(f"Done. Saved to {run_dir}")
    return run_dir
