#!/bin/bash
set -euo pipefail

PARTITION="atlasv2_mia_cpu01"

echo "Submitting YOLOv8n inference (CPU) to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  run_yolov8_inf_cpu.sbatch
