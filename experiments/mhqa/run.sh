#!/bin/bash

# Default values
models=("DeepSeek-V3" "gpt-4o-mini" "Llama-3.3-70B" "gpt-4o")
datasets=("HotpotQA" "2WikiMultihopQA" "MuSiQue")
prompts=("direct" "cot" "top-k" "multistep")
num_samples=1000
num_workers=10
temperature=0.7
debug=false

print_usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Run experiments with specified models, datasets, and prompts.

Options:
  --models MODEL1 [MODEL2 ...]       Specify models (default: ${models[*]})
  --datasets DATASET1 [DATASET2 ...] Specify datasets (default: ${datasets[*]})
  --prompts PROMPT1 [PROMPT2 ...]    Specify prompts (default: ${prompts[*]})
  --num_samples N                    Number of samples (default: $num_samples)
  --temperature T                    Temperature (default: $temperature)
  --num_workers N                  Number of workers (default: $num_workers)
  --debug                            Enable debug mode
  -h, --help                         Show this help message and exit

Example:
  $0 --models gpt-4o-mini --datasets HotpotQA 2WikiMultihopQA --prompts direct cot --num_samples 500
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
  --models)
    shift     # past the --models argument
    models=() # Clear default models
    while [[ $# -gt 0 && "$1" != --* ]]; do
      models+=("$1")
      shift
    done
    ;;
  --datasets)
    shift       # past the --datasets argument
    datasets=() # Clear default datasets
    while [[ $# -gt 0 && "$1" != --* ]]; do
      datasets+=("$1")
      shift
    done
    ;;
  --prompts)
    shift      # past the --prompts argument
    prompts=() # Clear default prompts
    while [[ $# -gt 0 && "$1" != --* ]]; do
      prompts+=("$1")
      shift
    done
    ;;
  --num_samples)
    shift
    num_samples="$1"
    shift
    ;;
  --num_workers)
    shift
    num_workers="$1"
    shift
    ;;
  --temperature)
    shift
    temperature="$1"
    shift
    ;;
  --debug)
    debug=true
    shift
    ;;
  -h | --help)
    print_usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    print_usage
    exit 1
    ;;
  esac
done

# Validate inputs
[[ ${#models[@]} -eq 0 ]] && echo "Error: No models specified." && exit 1
[[ ${#datasets[@]} -eq 0 ]] && echo "Error: No datasets specified." && exit 1
[[ ${#prompts[@]} -eq 0 ]] && echo "Error: No prompts specified." && exit 1

# Calculate total number of runs
total_runs=$((${#models[@]} * ${#datasets[@]} * ${#prompts[@]}))
current_run=0
total_time=0 # in seconds

echo "Total runs to complete: $total_runs"

# Run experiments
for dataset in "${datasets[@]}"; do
  for prompt in "${prompts[@]}"; do
    for model in "${models[@]}"; do
      # Increment run counter
      current_run=$((current_run + 1))

      # Echo progress
      echo "Progress: ${current_run}/${total_runs}"

      # Estimate time remaining (after the first run)
      if [ "$current_run" -gt 1 ]; then
        completed_runs=$((current_run - 1))
        average_time=$((total_time / completed_runs))
        remaining_runs=$((total_runs - current_run + 1))
        estimated_time=$((average_time * remaining_runs))
        echo "Estimated time remaining: $(date -u -d "@${estimated_time}" +%H:%M:%S)"
      fi

      echo "#### Running model: $model | dataset: $dataset | prompt: $prompt"
      args=(
        --model "$model"
        --dataset "$dataset"
        --prompt "$prompt"
        --num_samples "$num_samples"
        --num_workers "$num_workers"
        --temperature "$temperature"
        --project_name "Verbalized Multistep Confidence"
      )
      $debug && args+=(--debug)
      echo "Command: python run.py ${args[*]}"
      # Measure time for the current run
      start_time=$(date +%s)
      python run.py "${args[@]}"
      end_time=$(date +%s)
      duration=$((end_time - start_time))

      # Add duration to total time
      total_time=$((total_time + duration))

      # Check the exit status of the last command
      if [ $? -ne 0 ]; then
        echo "Error: Command failed for model: $model | dataset: $dataset | prompt: $prompt" >&2
        exit 1
      fi
    done
  done
done

echo "All runs completed."
