#!/bin/bash

perturbation_list=("invisible")
distribution_list=("uniform" "normal")
density_list=(0.05 0.2)
model_list=("bert-base-uncased")
dataset_list=("rotten_tomatoes")

for perturbation in "${perturbation_list[@]}"; do
  for distribution in "${distribution_list[@]}"; do
    for density in "${density_list[@]}"; do
      for model in "${model_list[@]}"; do
        for dataset in "${dataset_list[@]}"; do
          python adversarial_training.py --perturbation "$perturbation" --distribution "$distribution" --density "$density" --model "$model" --dataset "$dataset"
        done
      done
    done
  done
done
