import os
import json
import datetime
import subprocess
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_SYSTEM_PROMPT = """Answer the following multiple-choice question. Choose the correct answer from the choices provided.

Return EXACTLY this JSON:
{"answer": "...", "confidence": number}

Rules:
- answer must be the letter of the correct choice (A, B, C, D)
- confidence must be an integer between 0 and 100
- no extra text"""

def format_question(example):
    """Format a multiple-choice example into a string."""
    question = example['question']
    choices = example['choices']
    lines = [f"Question: {question}", "Choices:"]
    for label, text in zip(choices['label'], choices['text']):
        lines.append(f"{label}: {text}")
    return '\n'.join(lines)

def chunk_list(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def print_metadata(metadata):
    print(f"Running evaluation with model: {metadata['model']}")
    print(f"Batch size: {metadata['batch_size']}")
    print(f"System prompt: {metadata['system_prompt']}")
    print(f"Device: {metadata['device']}")
    if metadata['max_samples'] is not None:
        print(f"Max samples: {metadata['max_samples']}")
    if metadata['git_commit']:
        print(f"Git commit: {metadata['git_commit']}")

def compute_choice_log_probs(model, tokenizer, system_prompt, formatted_question, choices_text, device='cuda'):
    """
    Compute log probabilities for each choice text given the context.
    context = system_prompt + '\n\n' + formatted_question + '\nAnswer: '
    We compute log p(choice | context) using the model's next token predictions.
    Returns list of log probabilities (float) for each choice.
    """
    # Build context string
    context = system_prompt + '\n\n' + formatted_question + '\nAnswer: '
    # Tokenize context (without padding)
    context_inputs = tokenizer(context, return_tensors='pt').to(device)
    context_ids = context_inputs.input_ids[0]  # shape (context_len)
    context_len = context_ids.shape[0]
    
    # Tokenize each choice separately (without special tokens)
    choice_ids_list = []
    for choice in choices_text:
        choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors='pt').input_ids[0].to(device)
        choice_ids_list.append(choice_ids)
    
    # Create a batch where each sample is context + choice
    # We'll pad choices to max length
    max_choice_len = max(len(ids) for ids in choice_ids_list)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_input_ids = []
    batched_attention_mask = []
    for choice_ids in choice_ids_list:
        # Concatenate context_ids and choice_ids
        input_ids = torch.cat([context_ids, choice_ids])
        # Pad to max length (context_len + max_choice_len)
        padding_len = max_choice_len - len(choice_ids)
        if padding_len > 0:
            input_ids = torch.cat([input_ids, torch.full((padding_len,), pad_token_id, device=device)])
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.ones(context_len + len(choice_ids), device=device)
        if padding_len > 0:
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_len, device=device)])
        batched_input_ids.append(input_ids)
        batched_attention_mask.append(attention_mask)
    
    # Stack into tensors
    input_ids = torch.stack(batched_input_ids)  # shape (num_choices, seq_len)
    attention_mask = torch.stack(batched_attention_mask)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape (num_choices, seq_len, vocab_size)
        # Compute log probabilities for each choice token
        choice_log_probs = []
        for i, choice_ids in enumerate(choice_ids_list):
            # positions corresponding to choice tokens (after context)
            start_pos = context_len
            log_prob = 0.0
            # logits[i, pos-1] corresponds to prediction for token at position pos
            for idx, token_id in enumerate(choice_ids):
                pos = start_pos + idx
                # logits at pos-1
                token_logits = logits[i, pos - 1]  # shape (vocab_size)
                log_softmax = torch.log_softmax(token_logits, dim=-1)
                token_log_prob = log_softmax[token_id].item()
                log_prob += token_log_prob
            choice_log_probs.append(log_prob)
    return choice_log_probs

def run_ai2_arc(model_name, system_prompt, batch_size=4, out_dir="out_runs", use_cuda=False, max_samples=None, dataset_config="ARC-Challenge"):
    # Setup output directory
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Collect metadata
    metadata = {
        "model": model_name,
        "batch_size": batch_size,
        "system_prompt": system_prompt,
        "timestamp": run_timestamp,
        "git_commit": None,
        "device": "cuda" if use_cuda else "cpu",
        "max_samples": max_samples,
        "dataset_config": dataset_config
    }

    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        metadata["git_commit"] = commit_hash
    except Exception:
        metadata["git_commit"] = "unknown"

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print_metadata(metadata)

    # Load AI2 ARC dataset with specified config
    dataset = load_dataset("allenai/ai2_arc", dataset_config)
    test_data = dataset["test"]
    if max_samples is not None:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
        print(f"Debug mode: limiting to {len(test_data)} samples")

    # Save test questions with choices
    with open(os.path.join(run_dir, "test_questions.json"), "w") as f:
        json.dump([{'id': item['id'], 'question': item['question'], 'choices': item['choices']} for item in test_data], f, indent=2)

    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None if not use_cuda else 'auto', torch_dtype=torch.float16 if use_cuda else torch.float32)
    device = 'cuda' if use_cuda else 'cpu'
    model.to(device)
    model.eval()

    outputs_log = []

    for batch_idx, batch in enumerate(tqdm(list(chunk_list(test_data, batch_size)), desc="Batches")):
        # Prepare formatted questions for generation
        formatted_batch = [format_question(item) for item in batch]
        prompts = [system_prompt + '\n\n' + q + '\nAnswer: ' for q in formatted_batch]
        # Tokenize and generate
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
        # Decode generated answers
        generated_texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        # Remove prompts from generated texts
        generated_answers = []
        for i, text in enumerate(generated_texts):
            prompt = prompts[i]
            answer = text[len(prompt):].strip() if text.startswith(prompt) else text
            generated_answers.append(answer)

        # For each item in batch, compute choice log probs
        for i, item in enumerate(batch):
            formatted = formatted_batch[i]
            choices_text = item['choices']['text']
            choice_log_probs = compute_choice_log_probs(model, tokenizer, system_prompt, formatted, choices_text, device=device)
            
            outputs_log.append({
                'id': item['id'],
                'question': item['question'],
                'choices': item['choices'],
                'answerKey': item.get('answerKey', None),
                'generated_answer': generated_answers[i],
                'choice_log_probs': choice_log_probs,
                'choice_labels': item['choices']['label']
            })

    # Save outputs
    with open(os.path.join(run_dir, "outputs.json"), "w") as f:
        json.dump(outputs_log, f, indent=2)

    print(f"Run complete. All data saved in {run_dir}")
    return run_dir