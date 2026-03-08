#!/bin/bash
#SBATCH --job-name=yolov8-inf
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "YOLOv8n inference – GPU mode"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi

apptainer exec --nv --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    echo 'COMMAND: accelerate launch --num_machines=1 --num_processes=1 /accelerate_scripts/yolov8-inf_gpu.py'
    accelerate launch --num_machines=1 --num_processes=1 ${HOME}/accelerate_scripts/yolov8-inf_gpu.py
  "
