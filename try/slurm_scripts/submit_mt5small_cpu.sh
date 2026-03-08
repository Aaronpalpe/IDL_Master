#!/bin/bash
set -euo pipefail


# Default partition (single GPU)
PARTITION="atlasv2_mia_cpu01"


echo "Submitting mt5small CPU to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  run_mt5small_cpu.sbatch

