import argparse # Keep for eval function
# main llm inference
from tqdm import tqdm
import pandas as pd
from utils.processor import get_processor
from utils.chat import get_llm
from utils.datasets import get_dataset
import wandb
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.gpt_eval import gpt_eval
from eval import evaluate
from logistic_regression import logistic_regression

thread_local = threading.local()

# Main function to run the process
def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run inference and evaluation for LLM model."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., 'HotpotQA', '2WikiMultihopQA').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the language model to use (e.g., 'gpt-4o-mini', 'deepseek').",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompting strategy to apply ('direct', 'cot', 'topk', 'multistep').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to process from the dataset.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the language model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel workers for processing samples.",
    )
    parser.add_argument(
        "--project_name",
        required=True,
        help="Name of the Weights & Biases project to log results.",
    )

    args = parser.parse_args()
    print(args)

    run = wandb.init(
        project=args.project_name,
        job_type="inference",
        config=args,  # type: ignore
    )

    dataset = get_dataset(args.dataset, args.num_samples) # Assuming get_dataset returns a list of samples

    llm_config = {
        "model": args.model,
        "temperature": args.temperature,
    }

    processor = get_processor(args.prompt)


    def process_sample(sample):
        if not hasattr(thread_local, 'llm'):
            thread_local.llm = get_llm(**llm_config)
        sample = processor.process_sample(sample, thread_local.llm)
        return sample

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results_list = list(tqdm(executor.map(process_sample, dataset), total=len(dataset), desc="Answering Questions"))
        table = pd.DataFrame(results_list)

    # Evaluate all the successfully processed samples
    results = table[table["texts"].apply(lambda x: len(x) > 0)]
    results = pd.DataFrame(map(processor.eval_sample, results.to_dict(orient="records")))
    results = results.drop(columns=["texts", "confidences"], errors='ignore')

    # gpt eval
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        gpt_eval_results = list(tqdm(executor.map(lambda sample: gpt_eval(sample, thread_local), results.to_dict(orient="records")), total=len(results), desc="GPT Evals"))
        results["gpt_eval"] = gpt_eval_results


    failed = table[table["texts"].apply(lambda x: len(x) == 0)]
    failed = failed.drop(columns=["texts", "confidences"], errors='ignore')

    table = wandb.Table(dataframe=results)
    failed = wandb.Table(dataframe=failed)

    # Define the artifact name based on prompt, dataset, and model
    artifact_name = f"{args.prompt}-{args.dataset}-{args.model}"

    # Create the wandb.Artifact
    inference_artifact = wandb.Artifact(
        name=artifact_name,
        type="inference",        # Set artifact type to "inference"
        metadata=vars(args)      # Add run arguments as metadata
    )

    # Add the W&B tables to the artifact
    # The eval function expects the main results table to be named "table"
    inference_artifact.add(table, name="table")
    inference_artifact.add(failed, name="failed") # Add failed samples table

    # Log the artifact
    run.log_artifact(inference_artifact)

    # Keep the original direct logging of tables for easy viewing in the run's workspace
    wandb.log({"table": table, "failed": failed})

    wandb.finish() # Finish the main inference run



    ##### Start raw evaluation

    run = wandb.init(
        project=args.project_name,
        job_type="eval",
        config=args,  # type: ignore
    )

    table_artifact = run.use_artifact(
        f"{args.project_name}/{artifact_name}:latest"
    )
    table_df: pd.DataFrame = table_artifact.get("table").get_dataframe() # Renamed to table_df to avoid conflict
    run.config.update(table_artifact.logged_by().config) # type: ignore

    results_dict = evaluate(table_df["gpt_eval"].to_numpy(), table_df["confidence"].to_numpy())

    wandb.log(
        results_dict
    )

    run.finish()


    ##### start logistic regression
    run = wandb.init(
        project=args.project_name,
        job_type="logistic_regression",
        config=args,  # type: ignore
    )

    table_artifact = run.use_artifact(
        f"{args.project_name}/{artifact_name}:latest"
    )
    table_df: pd.DataFrame = table_artifact.get("table").get_dataframe() # Renamed to table_df to avoid conflict
    run.config.update(table_artifact.logged_by().config) # type: ignore

    # Prepare data for logistic regression
    y = table_df["gpt_eval"].to_numpy()
    if args.prompt == "multistep":
        # for multistep we use final_confidence, mean_confidence, min_confidence, max_confidence, num_steps
        X = table_df[["confidence", "final_confidence", "mean_confidence", "min_confidence", "max_confidence", "num_steps"]].to_numpy()
    else:
        X = table_df["confidence"].to_numpy().reshape(-1, 1)

    results_dict = logistic_regression(X, y)

    wandb.log(
        results_dict
    )
    run.finish()


# Entry point for script
if __name__ == "__main__":
    main()
