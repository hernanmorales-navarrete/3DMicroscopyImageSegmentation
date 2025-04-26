#!/bin/bash

# Exit on error
set -e

echo "Phase 1: Data Preparation and Training"
echo "======================================"

# Hardcoded dataset paths
declare -a DATASETS=(
    "data/processed/mouse"
    "data/processed/Sinusoids_filled"
    "data/processed/BC"
    "data/processed/Sinusoids"
)

# First phase: Prepare data and train models for all datasets
for dataset_dir in "${DATASETS[@]}"; do
    dataset_name=$(basename "${dataset_dir}")
    echo "Preparing data for dataset: ${dataset_name}"

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
    for model in "UNet3D" "AttentionUNet3D"; do
        echo "Training ${model} model..."
        for augmentation in "NONE" "STANDARD" "OURS"; do
            echo "Training with ${augmentation} augmentation..."
            if [ "${augmentation}" = "OURS" ]; then
                python src/modeling/train.py "${model}" "${training_dir}/patches" "${dataset_name}" --augmentation "${augmentation}" --psf "${psf_file}"
            else
                python src/modeling/train.py "${model}" "${training_dir}/patches" "${dataset_name}" --augmentation "${augmentation}"
            fi
        done
    done

    echo "Completed training for dataset: ${dataset_name}"
    echo "----------------------------------------"
done

echo "Phase 1 completed successfully!" 