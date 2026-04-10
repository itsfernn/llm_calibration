# run_gsm8k_cli.py
import argparse
from gsm8k import run_gsm8k, DEFAULT_SYSTEM_PROMPT as GSM8K_PROMPT
from ai2_arc import run_ai2_arc, DEFAULT_SYSTEM_PROMPT as AI2_ARC_PROMPT

# Default system prompts per dataset
DEFAULT_SYSTEM_PROMPTS = {
    "gsm8k": GSM8K_PROMPT,
    "ai2_arc": AI2_ARC_PROMPT,
}

def run_dataset(dataset_name, model_name, batch_size, system_prompt, out_dir, use_cuda, max_samples, dataset_config=None):
    if dataset_name == "gsm8k":
        return run_gsm8k(model_name, system_prompt, batch_size, out_dir, use_cuda, max_samples)
    elif dataset_name == "ai2_arc":
        return run_ai2_arc(model_name, system_prompt, batch_size, out_dir, use_cuda, max_samples, dataset_config=dataset_config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a dataset evaluation")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Name of the Hugging Face dataset")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration (e.g., ARC-Challenge, ARC-Easy)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--system_prompt", type=str, default=GSM8K_PROMPT, help="System prompt")
    parser.add_argument("--out_dir", type=str, default="out_runs", help="Output directory")
    parser.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for debugging")

    args = parser.parse_args()

    run_dir = run_dataset(
        dataset_name=args.dataset,
        model_name=args.model,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt,
        out_dir=args.out_dir,
        use_cuda=args.use_cuda,
        max_samples=args.max_samples,
        dataset_config=args.dataset_config
    )

    print(f"Finished run. Outputs saved in {run_dir}")

