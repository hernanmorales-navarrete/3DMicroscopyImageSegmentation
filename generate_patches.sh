#!/bin/bash

# Exit on error
set -e

echo "Patch Generation Script"
echo "======================"

# Hardcoded dataset paths
declare -a DATASETS=(
    "data/processed/microvascular"
    "data/processed/Sinusoids_filled"
    "data/processed/BC"
    "data/processed/Sinusoids"
    "data/processed/vessels"
)

# Generate patches for all datasets
for dataset_dir in "${DATASETS[@]}"; do
    dataset_name=$(basename "${dataset_dir}")
    echo "Processing dataset: ${dataset_name}"

    # Generate patches for training data (no padding needed)
    echo "Generating patches for ${dataset_name} training data..."
    training_dir="${dataset_dir}/training_data"
    python src/generate_patches.py "${training_dir}" False

    # Generate patches for test data (with padding for reconstruction)
    echo "Generating patches for ${dataset_name} test data..."
    test_dir="${dataset_dir}/test_data"
    python src/generate_patches.py "${test_dir}" True

    # Generate patches for test data (without padding for evaluation)
    echo "Generating patches for ${dataset_name} test data..."
    test_dir="${dataset_dir}/test_data"
    python src/generate_patches.py "${test_dir}" False

    echo "Completed patch generation for dataset: ${dataset_name}"
    echo "----------------------------------------"
done

echo "Patch generation completed successfully!" 