#!/bin/bash

perturbation_list=("deletion")
distribution_list=("uniform" "normal")
density_list=(0.05 0.2)

for perturbation in "${perturbation_list[@]}"; do
  for distribution in "${distribution_list[@]}"; do
    for density in "${density_list[@]}"; do
      python test_clean_model.py --perturbation "$perturbation" --distribution "$distribution" --density "$density"
    done
  done
done
