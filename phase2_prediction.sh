#!/bin/bash

# Exit on error
set -e

echo "Phase 2: Generating Predictions"
echo "=============================="

# Hardcoded dataset paths
declare -a DATASETS=(
    "data/processed/mouse"
    "data/processed/Sinusoids_filled"
    "data/processed/BC"
    "data/processed/Sinusoids"
)

# Second phase: Generate predictions for all datasets
for dataset_dir in "${DATASETS[@]}"; do
    dataset_name=$(basename "${dataset_dir}")
    test_dir="${dataset_dir}/test_data"
    
    echo "Generating predictions for ${dataset_name} test data..."
    python src/modeling/predict.py "${test_dir}/padded_patches" "${test_dir}" "${dataset_name}"
    
    echo "Completed predictions for dataset: ${dataset_name}"
    echo "----------------------------------------"
done

echo "Phase 2 completed successfully!" 