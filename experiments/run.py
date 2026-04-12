# run_gsm8k_cli.py
import argparse
from gsm8k import run_gsm8k
from ai2_arc import run_ai2_arc

def run_dataset(dataset_name, **kwargs):
    if dataset_name == "gsm8k":
        return run_gsm8k(**kwargs)
    elif dataset_name == "ai2_arc":
        return run_ai2_arc(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a dataset evaluation")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Name of the Hugging Face dataset")
    parser.add_argument("--dataset_config", type=str, default=argparse.SUPPRESS, help="Dataset configuration (e.g., ARC-Challenge, ARC-Easy)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="out_runs", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for debugging")
    parser.add_argument("--no-thinking", action="store_false", dest="thinking", help="Whether to include step-by-step reasoning in the prompt")

    args = parser.parse_args()


    kwargs = vars(args)
    print(kwargs)
    run_dataset(args.dataset, **kwargs)

    print(f"Finished run. Outputs saved in {args.out_dir}")

