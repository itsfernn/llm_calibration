# run_gsm8k_cli.py
import argparse
from gsm8k import run_gsm8k
from gsm8k_fewshot import run_gsm8k_fewshot
from ai2_arc import run_ai2_arc
from mmlu import run_mmlu


def run_dataset(dataset_name, **kwargs):
    if dataset_name == "gsm8k":
        return run_gsm8k(**kwargs)
    elif dataset_name == "gsm8k_fewshot":
        return run_gsm8k_fewshot(**kwargs)
    elif dataset_name == "ai2_arc":
        return run_ai2_arc(**kwargs)
    elif dataset_name == "mmlu":
        return run_mmlu(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a dataset evaluation")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", help="Name of the Hugging Face dataset"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=argparse.SUPPRESS,
        help="Dataset configuration (e.g., ARC-Challenge, ARC-Easy)",
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--out_dir", type=str, default="out_runs", help="Output directory"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples for debugging",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_false",
        dest="thinking",
        help="Whether to include step-by-step reasoning in the prompt",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=3,
        help="Number of few-shot exemplars (gsm8k_fewshot only)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="gsm8k_fewshot: suppress chain-of-thought (post-trained models "
        "otherwise leave the Answer/Confidence fields empty)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags written to metadata.json (e.g. fewshot_comp)",
    )
    parser.add_argument(
        "--max_thinking_tokens",
        type=int,
        default=1000,
        help="Maximum tokens for the thinking phase (default: 1000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (prints additional information)",
    )

    args = parser.parse_args()

    kwargs = vars(args)
    print(kwargs)
    run_dataset(args.dataset, **kwargs)

    print(f"Finished run. Outputs saved in {args.out_dir}")
