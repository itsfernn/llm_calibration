#!/bin/bash

models=("gpt-4o-mini")
datasets=("2WikiMultihopQA" "MuSiQue")
prompts=("direct" "cot" "top-k" "multistep") #("direct" "cot")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for prompt in "${prompts[@]}"; do
      echo "          #### Running model: $model, dataset: $dataset, prompt: $prompt"
      # print command
      arguments=(
        --model "$model"
        --dataset "$dataset"
        --prompt "$prompt"
        --num_samples 1000
      )
      echo "Arguments: ${arguments[*]}"
      python run.py "${arguments[@]}"
    done
  done
done
