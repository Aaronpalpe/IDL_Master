#!/bin/bash
set -euo pipefail

PARTITION="atlasv2_mia_gpu01_1t4"

echo "Submitting YOLOv8n inference (GPU) to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  --gres="gpu:1" \
  run_yolov8_inf_gpu.sbatch
