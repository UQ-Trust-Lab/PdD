#!/bin/bash

perturbation_list=("deletion")
distribution_list=("uniform" "normal")
density_list=(0.01 0.1 0.3)
num_of_perturbations=(5 10)

for perturbation in "${perturbation_list[@]}"; do
  for distribution in "${distribution_list[@]}"; do
    for density in "${density_list[@]}"; do
      for num_of_perturbation in "${num_of_perturbations[@]}"; do
        python bert-based-uncased-rotten-tomatoes-deletion.py --perturbation "$perturbation" --distribution "$distribution" --density "$density" --num_of_perturbation "$num_of_perturbation"
      done
    done
  done
done
