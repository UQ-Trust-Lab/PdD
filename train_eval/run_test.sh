#!/bin/bash

perturbation_list=("deletion")
distribution_list=("uniform" "normal")
density_list=(0.01 0.1 0.3)

for perturbation in "${perturbation_list[@]}"; do
  for distribution in "${distribution_list[@]}"; do
    for density in "${density_list[@]}"; do
      python bert-based-uncased-rotten-tomatoes-deletion.py --perturbation "$perturbation" --distribution "$distribution" --density "$density"
    done
  done
done