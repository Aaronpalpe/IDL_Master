#!/bin/bash
set -euo pipefail


# Default partition (single GPU)
PARTITION="atlasv2_mia_cpu01"


echo "Submitting YOLOv8n CPU to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  run_yolov8_inf_cpu.sbatch
