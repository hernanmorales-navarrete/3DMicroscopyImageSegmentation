#!/bin/bash

# Exit on error
set -e

echo "Starting pipeline..."

# Hardcoded dataset paths
declare -a DATASETS=(
    "data/processed/mouse"
    "data/processed/Sinusoids_filled"
    "data/processed/BC"
    "data/processed/Sinusoids"
)

# Process each dataset
for dataset_dir in "${DATASETS[@]}"; do
    dataset_name=$(basename "${dataset_dir}")
    echo "Processing dataset: ${dataset_name}"

    # Select correct PSF file based on dataset
    if [ "${dataset_name}" = "mouse" ]; then
        psf_file="data/external/PSF_mouse.tif"
    else
        psf_file="data/external/PSF.tif"
    fi

    # Generate patches for training data (no padding needed)
    echo "Generating patches for ${dataset_name} training data..."
    training_dir="${dataset_dir}/training_data"
    python src/generate_patches.py "${training_dir}" False

    # Generate patches for test data (with padding for reconstruction)
    echo "Generating patches for ${dataset_name} test data..."
    test_dir="${dataset_dir}/test_data"
    python src/generate_patches.py "${test_dir}" True

    # Train models using training data
    echo "Training models for ${dataset_name}..."
    for augmentation in "NONE" "STANDARD" "OURS"; do
        echo "Training with ${augmentation} augmentation..."
        if [ "${augmentation}" = "OURS" ]; then
            python src/modeling/train.py "UNet" "${training_dir}/patches" --augmentation "${augmentation}" --psf "${psf_file}"
        else
            python src/modeling/train.py "UNet" "${training_dir}/patches" --augmentation "${augmentation}"
        fi
    done

    # Generate predictions on test data
    echo "Generating predictions for ${dataset_name} test data..."
    python src/modeling/predict.py "${test_dir}/patches" "${test_dir}" "models" "${dataset_name}"

    # Generate evaluation plots using test data
    echo "Generating evaluation plots for ${dataset_name}..."
    python src/plots.py "${test_dir}/patches" "${test_dir}" "models"

    echo "Completed processing dataset: ${dataset_name}"
    echo "----------------------------------------"
done

echo "Pipeline completed successfully!" 