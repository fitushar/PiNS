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
docker exec nodule_seg_pipeline mkdir -p /app/demofolder/output/DLCS24_KNN_2mm_Extend_Seg
docker exec nodule_seg_pipeline chmod -R 777 /app/demofolder/output/

echo "Installing missing Python packages if needed..."
docker exec nodule_seg_pipeline pip install opencv-python-headless --quiet > /dev/null 2>&1 || true

echo "Docker container is running with write permissions set"

# ============================
# Configuration Variables  
# ============================
# Define paths
PYTHON_SCRIPT="/app/scr/candidateSeg_pipiline.py"  # Path inside container
PARAMS_JSON="/app/scr/Pyradiomics_feature_extarctor_pram.json"  # Path inside container

DATASET_NAME="DLCSD24"
RAW_DATA_PATH="/app/demofolder/data/DLCS24/"
CSV_SAVE_PATH="/app/demofolder/output/"
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

SEG_ALG="knn"  # Choose from: gmm, knn, fcm, otsu
EXPANSION_MM=2.0  # Set the expansion in millimeters
SAVE_NIFTI_PATH="/app/demofolder/output/DLCS24_KNN_2mm_Extend_Seg/"
SAVE_MASK_FLAG="--save_the_generated_mask"  # Remove if you don't want to save masks
USE_EXPAND_FLAG="--use_expand"  # Include if you want to use expansion
EXTRACT_RADIOMICS_FLAG=""  # Include if you want to extract radiomics "--extract_radiomics"

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
  --w "$W" \
  --h "$H" \
  --d "$D" \
  --seg_alg "$SEG_ALG" \
  --expansion_mm "$EXPANSION_MM" \
  --params_json "$PARAMS_JSON" \
  --save_nifti_path "$SAVE_NIFTI_PATH" \
  $USE_EXPAND_FLAG \
  $EXTRACT_RADIOMICS_FLAG \
  $SAVE_MASK_FLAG

echo "✅ Segmentation completed! Check demofolder/output/ directory for results."