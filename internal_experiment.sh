#!/bin/bash

# Exit on error
set -e

echo "Starting pipeline..."

# Make phase scripts executable
chmod +x phase1_training.sh
chmod +x phase2_prediction.sh
chmod +x phase3_evaluation.sh

# Run each phase
./phase1_training.sh
./phase2_prediction.sh
./phase3_evaluation.sh

echo "Pipeline completed successfully!" 