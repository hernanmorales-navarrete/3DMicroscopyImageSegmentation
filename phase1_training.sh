#!/bin/bash

# Exit on error
set -e

echo "Phase 1: Data Preparation and Training"
echo "======================================"

# Hardcoded dataset paths
declare -a DATASETS=(
    "data/processed/Sinusoids_filled"
    "data/processed/BC"
    "data/processed/Sinusoids"
    "data/processed/vessels"
)

# First phase: Prepare data and train models for all datasets
for dataset_dir in "${DATASETS[@]}"; do
    dataset_name=$(basename "${dataset_dir}")
    echo "Preparing data for dataset: ${dataset_name}"
    training_dir="${dataset_dir}/training_data"

    # Select correct PSF file based on dataset
    if [ "${dataset_name}" = "mouse" ]; then
        psf_file="data/external/PSF_mouse.tif"
    else
        psf_file="data/external/PSF.tif"
    fi

    # Train models using training data
    echo "Training models for ${dataset_name}..."
    for model in "UNet3D" "AttentionUNet3D"; do
        echo "Training ${model} model..."
        for augmentation in "NONE" "STANDARD" "OURS"; do
            echo "Training with ${augmentation} augmentation..."
            if [ "${augmentation}" = "OURS" ]; then
                python src/modeling/train.py "${model}" "${training_dir}/regular_patches" "${dataset_name}" --augmentation "${augmentation}" --psf "${psf_file}"
            else
                python src/modeling/train.py "${model}" "${training_dir}/regular_patches" "${dataset_name}" --augmentation "${augmentation}"
            fi
        done
    done

    echo "Completed training for dataset: ${dataset_name}"
    echo "----------------------------------------"
done

echo "Phase 1 completed successfully!" 
