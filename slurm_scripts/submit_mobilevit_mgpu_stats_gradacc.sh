#!/bin/bash
set -euo pipefail

NUM_PROCESSES="${NUM_GPUs:-1}"

# Default partition (single GPU)
PARTITION_DEFAULT="atlasv2_mia_gpu01_1t4"
# Multi-GPU partition
PARTITION_MULTI="atlasv2_mia_gpu02_4t4"

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  PARTITION="${PARTITION_MULTI}"
else
  PARTITION="${PARTITION_DEFAULT}"
fi

echo "Submitting with NUM_GPUs=${NUM_PROCESSES} to partition=${PARTITION} (stats + gradient accumulation)"

sbatch \
  --partition="${PARTITION}" \
  --gres="gpu:${NUM_PROCESSES}" \
  --export=ALL,NUM_PROCESSES="${NUM_PROCESSES}" \
  run_mobilevit_mgpu_stats_gradacc.sbatch
