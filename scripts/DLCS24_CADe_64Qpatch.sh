#!/bin/bash

# ============================
# Docker Container Activation
# ============================
echo "Starting Docker container..."
cd "$(dirname "$0")/.."  # Go to project root

# Remove existing container if it exists
docker rm -f nodule_seg_pipeline 2>/dev/null || true

# Start container using existing medical imaging image
docker run -d --name nodule_seg_pipeline \
  -v "$(pwd):/app" \
  -w /app \
  ft42/pins:latest \
  tail -f /dev/null

# Create output directory and set proper permissions
echo "Setting up output directories and permissions..."
docker exec nodule_seg_pipeline mkdir -p /app/demofolder/output/DLCS24_64Q_CAD_patches
docker exec nodule_seg_pipeline chmod -R 777 /app/demofolder/output/

echo "Installing missing Python packages..."
docker exec nodule_seg_pipeline apt-get update > /dev/null 2>&1
docker exec nodule_seg_pipeline apt-get install -y libgl1 libglib2.0-0 > /dev/null 2>&1
docker exec nodule_seg_pipeline pip install opencv-python-headless torch torchvision monai "numpy<2.0" --quiet

echo "Docker container is running with all dependencies installed"

# ============================
# Configuration Variables  
# ============================
# Define paths
PYTHON_SCRIPT="/app/scr/candidate_worldCoord_patchExtarctor_pipeline.py"  # Path inside container


DATASET_NAME="DLCSD24"
RAW_DATA_PATH="/app/demofolder/data/DLCS24/"
CSV_SAVE_PATH="/app/demofolder/output/DLCS24_64Q_CAD_patches/"
DATASET_CSV="/app/demofolder/data/DLCSD24_Annotations_N2.csv"

NIFTI_CLM_NAME="ct_nifti_file"
UNIQUE_ANNOTATION_ID="nodule_id"  # Leave empty or remove if not in CSV
MALIGNANT_LBL="Malignant_lbl"
COORD_X="coordX"
COORD_Y="coordY"
COORD_Z="coordZ"
W="w"
H="h"
D="d"
SAVE_NIFTI_PATH="/app/demofolder/output/DLCS24_64Q_CAD_patches/nifti/"
PATCH_SIZE="64 64 64"
NORMALIZATION="-1000 500 0 1"
CLIP="True"  # Change to "False" if needed

# ============================
# Run the Python script in Docker
# ============================
echo "Running segmentation in Docker container..."
docker exec nodule_seg_pipeline python3 "$PYTHON_SCRIPT" \
  --dataset_name "$DATASET_NAME" \
  --raw_data_path "$RAW_DATA_PATH" \
  --csv_save_path "$CSV_SAVE_PATH" \
  --dataset_csv "$DATASET_CSV" \
  --nifti_clm_name "$NIFTI_CLM_NAME" \
  --unique_Annotation_id "$UNIQUE_ANNOTATION_ID" \
  --Malignant_lbl "$MALIGNANT_LBL" \
  --coordX "$COORD_X" \
  --coordY "$COORD_Y" \
  --coordZ "$COORD_Z" \
  --patch_size $PATCH_SIZE \
  --normalization $NORMALIZATION \
  --clip "$CLIP" \
  --save_nifti_path "$SAVE_NIFTI_PATH"

