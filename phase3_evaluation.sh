#!/bin/bash

# Exit on error
set -e

echo "Phase 3: Generating Evaluation Plots"
echo "==================================="

# Hardcoded dataset paths
declare -a DATASETS=(
    "data/processed/microvascular"
    "data/processed/Sinusoids_filled"
    "data/processed/BC"
    "data/processed/Sinusoids"
)

# Third phase: Generate evaluation plots for all datasets
for dataset_dir in "${DATASETS[@]}"; do
    dataset_name=$(basename "${dataset_dir}")
    test_dir="${dataset_dir}/test_data"
    
    echo "Generating evaluation plots for ${dataset_name}..."
    python src/plots.py "${test_dir}/reconstruction_patches" "${test_dir}/regular_patches" "${test_dir}" "${dataset_name}"
    
    echo "Completed evaluation for dataset: ${dataset_name}"
    echo "----------------------------------------"
done

echo "Phase 3 completed successfully!" 