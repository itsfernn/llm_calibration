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

THINKING_PROMPT = """You are an expert mathematical assistant. 

Instructions:
- First solve the problem step-by-step.
- Once your thoughts are complete, provide your final response.
- Estimate your confidence in the correctness of your answer as a float between 0 and 1.
  - 0.0 = pure guess
  - 1.0 = absolutely certain

Your final response should ONLY valid JSON. No extra text!

Format:
{
  "answer": <final numeric answer>,
  "confidence": <float>
}"""


NON_THINKING_PROMPT = """You are a direct mathematical calculator.

Instructions:
- Solve the problem directly.
- Estimate your confidence in the correctness of your answer as a float between 0 and 1.
  - 0.0 = pure guess
  - 1.0 = absolutely certain

Output ONLY valid JSON. No extra text!

Format:
{
  "answer": <final numeric answer>,
  "confidence": <float>
}"""


def score_sequences(model, inputs):
    """
    Compute logprobs for every token in the sequence (prompt + generation).
    Returns a token_logprobs tensor of shape (batch, seq_len - 1)
    """

    with torch.inference_mode():
        # FIX 1: Add .logits to capture the raw tensor from the output object
        outputs = model(**inputs)
        logits = outputs.logits

    # Standard causal alignment shift
    # Logit at index t predicts token at index t + 1
    shift_logits = logits[:, :-1, :]
    shift_targets = inputs["input_ids"][:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)

    return token_logprobs


def run_gsm8k(
    model,
    batch_size,
    thinking,
    out_dir="out_runs",
    max_samples=None,
    system_prompt=None,
    debug=False,
    max_thinking_tokens=1000,
    **kwargs,
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

    # set system prompt if not provided
    if system_prompt is None:
        system_prompt = THINKING_PROMPT if thinking else NON_THINKING_PROMPT

    # Metadata
    metadata = {
        "model": model,
        "batch_size": batch_size,
        "thinking": thinking,
        "system_prompt": system_prompt,
        "timestamp": run_timestamp,
        "device": str(device),
        "max_samples": max_samples,
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

    def preprocess_batch(examples):
        base_texts = []
        for q in examples["question"]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
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
            # If not thinking, we can directly append the answer prefix to the base texts and tokenize
            full_texts = [t + ANSWER_PREFIX for t in base_texts]
            tokenized = tokenizer(full_texts, padding=False, truncation=True)
            out_dict["input_ids"] = tokenized["input_ids"]
            out_dict["attention_mask"] = tokenized["attention_mask"]
        else:
            # Tokenize base texts for thinking step
            tokenized_base = tokenizer(base_texts, padding=False, truncation=True)
            out_dict["base_input_ids"] = tokenized_base["input_ids"]
            out_dict["base_attention_mask"] = tokenized_base["attention_mask"]
        return out_dict

    test_data = test_data.map(preprocess_batch, batched=True, batch_size=batch_size)

    output_path = os.path.join(run_dir, "outputs.jsonl")
    f_out = open(output_path, "w")

    for start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
        end = min(start + batch_size, len(test_data))
        batch = test_data.select(range(start, end))

        if thinking:
            batch_features = [
                {
                    "input_ids": b["base_input_ids"],
                    "attention_mask": b["base_attention_mask"],
                }
                for b in batch
            ]

            thinking_inputs = tokenizer.pad(
                batch_features, return_tensors="pt", padding=True
            ).to(device)

            with torch.inference_mode():
                out = model.generate(
                    **thinking_inputs,
                    eos_token_id=END_THINK_TOKEN_ID,
                    max_new_tokens=max_thinking_tokens,
                    **gen_kwargs,
                )

            clean_input_ids = []

            for seq in out:
                clean_seq = seq[seq != tokenizer.pad_token_id].tolist()

                if clean_seq and clean_seq[-1] != END_THINK_TOKEN_ID:
                    clean_seq.append(END_THINK_TOKEN_ID)

                clean_seq.extend(answer_prefix_ids)

                clean_input_ids.append(clean_seq)

            final_inputs = tokenizer.pad(
                [{"input_ids": seq} for seq in clean_input_ids],
                return_tensors="pt",
                padding="longest",
            ).to(device)

            thinking_ids = out[:, thinking_inputs["input_ids"].shape[1] :]
            thinking_texts = tokenizer.batch_decode(
                thinking_ids, skip_special_tokens=True
            )

        else:
            batch_features = [
                {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                for b in batch
            ]
            final_inputs = tokenizer.pad(
                batch_features, return_tensors="pt", padding=True
            ).to(device)
            thinking_texts = None

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

            output = parse_output(content)
            prediction = output["answer"]
            confidence = output["confidence"]

            prediction_logprobs = []
            prediction_tokens = []

            if prediction is not None:
                # Find where the prediction string lives inside the decoded content
                start_idx = content.find(prediction)

                if start_idx != -1:
                    end_idx = start_idx + len(prediction)
                    current_len = 0

                    # Step through tokens progressively
                    for step_idx, token_id in enumerate(content_ids):
                        if current_len >= end_idx:
                            break

                        extended_text = tokenizer.decode(
                            content_ids[: step_idx + 1], skip_special_tokens=True
                        )
                        next_len = len(extended_text)

                        if next_len > start_idx and current_len < end_idx:
                            prediction_tokens.append(content_ids[step_idx])

                            global_token_idx = input_len + step_idx

                            if global_token_idx > 0:
                                lp = log_probs[i, global_token_idx - 1].item()
                                prediction_logprobs.append(lp)

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
                "thinking": thinking_texts[i] if thinking else None,
                "prediction": prediction,
                "logprobs": prediction_logprobs,
                "verb_conf": confidence,
            }
            outputs.append(output)

        if debug:
            for o in outputs:
                print(f"--- Sample {o['index']} ---")
                print(f"Question: {o['question']}")
                print(f"Ground truth: {o['answer']}")
                if thinking:
                    print(f"Thinking: {o['thinking']}")
                print(f"Generated: {o['content']}")
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
