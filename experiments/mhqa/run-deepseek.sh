#!/bin/bash

models=("DeepSeek-V3" "Llama-3.3-70B") #"gpt-4o-mini" "gpt-4o"
datasets=("2WikiMultihopQA")
prompts=("top-k" "multistep") #("direct" "cot")

for dataset in "${datasets[@]}"; do
  for prompt in "${prompts[@]}"; do
    for model in "${models[@]}"; do
      echo ""
      echo "          #### Running model: $model, dataset: $dataset, prompt: $prompt"
      echo ""
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

models=("DeepSeek-V3" "Llama-3.3-70B") #"gpt-4o-mini" "gpt-4o"
datasets=("MuSiQue")
prompts=("direct" "cot" "top-k" "multistep") #("direct" "cot")

for dataset in "${datasets[@]}"; do
  for prompt in "${prompts[@]}"; do
    for model in "${models[@]}"; do
      echo ""
      echo "          #### Running model: $model, dataset: $dataset, prompt: $prompt"
      echo ""
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
