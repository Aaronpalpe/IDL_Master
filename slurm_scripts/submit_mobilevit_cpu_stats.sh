#!/bin/bash
set -euo pipefail

# Default CPU partition
PARTITION="atlasv2_mia_cpu01"

echo "Submitting to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  run_mobilevit_cpu_stats.sbatch
