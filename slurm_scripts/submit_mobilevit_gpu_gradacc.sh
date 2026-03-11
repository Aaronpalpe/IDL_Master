#!/bin/bash
set -euo pipefail

NUM_PROCESSES="${NUM_GPUs:-1}"

# Default partition (single GPU)
PARTITION_DEFAULT="atlasv2_mia_gpu01_1t4"

PARTITION="${PARTITION_DEFAULT}"

echo "Submitting with NUM_GPUs=${NUM_PROCESSES} to partition=${PARTITION} (gradient accumulation)"

sbatch \
  --partition="${PARTITION}" \
  --gres="gpu:${NUM_PROCESSES}" \
  --export=ALL,NUM_PROCESSES="${NUM_PROCESSES}" \
  run_mobilevit_gpu_gradacc.sbatch
