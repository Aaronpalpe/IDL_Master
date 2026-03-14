#!/bin/bash
set -euo pipefail

##############################################################################
# bench_submit.sh — Submit a SLURM job N times and collect performance metrics
#
# Usage:
#   ./bench_submit.sh -n <runs> -l <label> -d <device> [-p <poll_seconds>] [-s] <job.sbatch> [-- extra sbatch args...]
#
# Options:
#   -n <runs>          Number of times to submit the job (required)
#   -l <label>         Experiment label for the output CSV (required)
#   -d <device>        Device mode: cpu, gpu, or mgpu (required)
#                        cpu  → partition=atlasv2_mia_cpu01
#                        gpu  → partition=atlasv2_mia_gpu01_1t4, gres=gpu:1
#                        mgpu → partition=atlasv2_mia_gpu02_4t4, gres=gpu:4
#   -p <seconds>       Polling interval in seconds (default: 30)
#   -s                 Submit sequentially — wait for each job to finish
#                      before submitting the next (default: submit all at once)
#   --                 Everything after this is passed as extra args to sbatch
#
# Output:
#   ~/bench/results/<label>_<YYYYMMDD_HHMMSS>.csv
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── Defaults ────────────────────────────────────────────────────────────────
NUM_RUNS=""
LABEL=""
DEVICE=""
POLL_INTERVAL=30
SEQUENTIAL=0
SBATCH_FILE=""
EXTRA_ARGS=()

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) NUM_RUNS="$2"; shift 2 ;;
    -l) LABEL="$2"; shift 2 ;;
    -d) DEVICE="$2"; shift 2 ;;
    -p) POLL_INTERVAL="$2"; shift 2 ;;
    -s) SEQUENTIAL=1; shift ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -*)
      echo "Error: Unknown option '$1'" >&2
      echo "Usage: $0 -n <runs> -l <label> -d <cpu|gpu|mgpu> [-p <poll_sec>] [-s] <job.sbatch> [-- extra sbatch args]" >&2
      exit 1
      ;;
    *)
      SBATCH_FILE="$1"; shift ;;
  esac
done

# ── Validate ────────────────────────────────────────────────────────────────
if [[ -z "${NUM_RUNS}" || -z "${LABEL}" || -z "${DEVICE}" || -z "${SBATCH_FILE}" ]]; then
  echo "Error: -n <runs>, -l <label>, -d <cpu|gpu|mgpu>, and <job.sbatch> are required." >&2
  echo "Usage: $0 -n <runs> -l <label> -d <cpu|gpu|mgpu> [-p <poll_sec>] [-s] <job.sbatch> [-- extra sbatch args]" >&2
  exit 1
fi

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "Error: File '${SBATCH_FILE}' not found." >&2
  exit 1
fi

# ── Device mode → partition & sbatch flags ─────────────────────────────────
PARTITION=""
DEVICE_ARGS=()

case "${DEVICE}" in
  cpu)
    PARTITION="atlasv2_mia_cpu01"
    ;;
  gpu)
    PARTITION="atlasv2_mia_gpu01_1t4"
    DEVICE_ARGS=(--gres="gpu:1" --export="ALL,NUM_PROCESSES=1")
    ;;
  mgpu)
    PARTITION="atlasv2_mia_gpu02_4t4"
    DEVICE_ARGS=(--gres="gpu:4" --export="ALL,NUM_PROCESSES=4")
    ;;
  *)
    echo "Error: Invalid device mode '${DEVICE}'. Must be cpu, gpu, or mgpu." >&2
    exit 1
    ;;
esac

# ── Helper: convert sacct time formats to seconds ──────────────────────────
# Handles: MM:SS.mmm, HH:MM:SS, D-HH:MM:SS
time_to_seconds() {
  local t="$1"
  if [[ -z "$t" || "$t" == "" ]]; then
    echo "0"
    return
  fi
  # Strip any trailing microsecond/millisecond part after a dot for non-elapsed fields
  local days=0 hours=0 minutes=0 seconds=0

  # D-HH:MM:SS format
  if [[ "$t" == *-* ]]; then
    days="${t%%-*}"
    t="${t#*-}"
  fi

  # Count colons to determine format
  local ncolons
  ncolons=$(echo "$t" | tr -cd ':' | wc -c)

  if [[ "$ncolons" -eq 2 ]]; then
    # HH:MM:SS or HH:MM:SS.mmm
    hours=$(echo "$t" | cut -d: -f1)
    minutes=$(echo "$t" | cut -d: -f2)
    seconds=$(echo "$t" | cut -d: -f3)
  elif [[ "$ncolons" -eq 1 ]]; then
    # MM:SS or MM:SS.mmm
    minutes=$(echo "$t" | cut -d: -f1)
    seconds=$(echo "$t" | cut -d: -f2)
  else
    seconds="$t"
  fi

  # Use awk for floating-point arithmetic
  echo "$days $hours $minutes $seconds" | awk '{printf "%.3f", $1*86400 + $2*3600 + $3*60 + $4}'
}

# ── Helper: convert sacct memory string (e.g. "123456K") to KB ─────────────
mem_to_kb() {
  local m="$1"
  if [[ -z "$m" || "$m" == "" ]]; then
    echo "0"
    return
  fi
  local unit="${m: -1}"
  local val="${m%?}"
  case "$unit" in
    K|k) echo "$val" ;;
    M|m) echo "$val" | awk '{printf "%.0f", $1 * 1024}' ;;
    G|g) echo "$val" | awk '{printf "%.0f", $1 * 1048576}' ;;
    *)   echo "$m" ;;  # already numeric (bytes) or unknown
  esac
}

# ── Submit jobs ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="${RESULTS_DIR}/${LABEL}_${TIMESTAMP}.csv"
JOB_IDS=()

echo "============================================================"
echo "Benchmark: ${LABEL}"
echo "Script:    ${SBATCH_FILE}"
echo "Device:    ${DEVICE} (partition=${PARTITION})"
echo "Runs:      ${NUM_RUNS}"
echo "Mode:      $([ "$SEQUENTIAL" -eq 1 ] && echo "sequential" || echo "parallel")"
echo "Output:    ${CSV_FILE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}"
fi
echo "============================================================"
echo ""

submit_one_job() {
  local run_num="$1"
  local output
  output=$(sbatch --partition="${PARTITION}" \
    "${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    "${SBATCH_FILE}" 2>&1)
  local jobid
  jobid=$(echo "$output" | grep -oP '\d+$')
  if [[ -z "$jobid" ]]; then
    echo "  [Run ${run_num}] ERROR: sbatch failed: ${output}" >&2
    return 1
  fi
  echo "  [Run ${run_num}] Submitted job ${jobid}"
  echo "$jobid"
}

wait_for_jobs() {
  local ids=("$@")
  local id_list
  id_list=$(IFS=,; echo "${ids[*]}")

  echo ""
  echo "Waiting for ${#ids[@]} job(s) to complete (polling every ${POLL_INTERVAL}s)..."

  while true; do
    local still_running=0
    for jid in "${ids[@]}"; do
      local state
      state=$(sacct -j "$jid" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
      if [[ "$state" == "PENDING" || "$state" == "RUNNING" || "$state" == "CONFIGURING" || "$state" == "COMPLETING" ]]; then
        still_running=$((still_running + 1))
      fi
    done

    if [[ "$still_running" -eq 0 ]]; then
      echo "All jobs finished."
      break
    fi

    echo "  ${still_running} job(s) still running..."
    sleep "${POLL_INTERVAL}"
  done
}

if [[ "$SEQUENTIAL" -eq 1 ]]; then
  for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "── Run ${i}/${NUM_RUNS} ──"
    jobid=$(submit_one_job "$i" | tail -1)
    JOB_IDS+=("$jobid")
    wait_for_jobs "$jobid"
  done
else
  echo "Submitting ${NUM_RUNS} jobs..."
  for ((i = 1; i <= NUM_RUNS; i++)); do
    jobid=$(submit_one_job "$i" | tail -1)
    JOB_IDS+=("$jobid")
  done
  wait_for_jobs "${JOB_IDS[@]}"
fi

# ── Collect metrics via sacct ───────────────────────────────────────────────
echo ""
echo "Collecting metrics from sacct..."

# CSV header
echo "job_id,run,elapsed_s,total_cpu_s,user_cpu_s,system_cpu_s,max_rss_kb,max_vmsize_kb,ave_rss_kb,state,exit_code" > "${CSV_FILE}"

for idx in "${!JOB_IDS[@]}"; do
  jid="${JOB_IDS[$idx]}"
  run_num=$((idx + 1))

  # Query sacct for the batch step (which has the actual resource usage)
  # Try .batch first, fall back to the main job line
  sacct_line=$(sacct -j "$jid.batch" \
    --format=ElapsedRaw,TotalCPU,UserCPU,SystemCPU,MaxRSS,MaxVMSize,AveRSS,State,ExitCode \
    --noheader --parsable2 2>/dev/null | head -1)

  if [[ -z "$sacct_line" ]]; then
    # Fallback: use the main job record
    sacct_line=$(sacct -j "$jid" \
      --format=ElapsedRaw,TotalCPU,UserCPU,SystemCPU,MaxRSS,MaxVMSize,AveRSS,State,ExitCode \
      --noheader --parsable2 2>/dev/null | head -1)
  fi

  if [[ -z "$sacct_line" ]]; then
    echo "  [Job ${jid}] WARNING: No sacct data found, skipping."
    echo "${jid},${run_num},,,,,,,,,UNKNOWN," >> "${CSV_FILE}"
    continue
  fi

  IFS='|' read -r elapsed_raw total_cpu user_cpu system_cpu max_rss max_vmsize ave_rss state exit_code <<< "$sacct_line"

  # Convert times to seconds
  elapsed_s="${elapsed_raw}"  # ElapsedRaw is already in seconds
  total_cpu_s=$(time_to_seconds "$total_cpu")
  user_cpu_s=$(time_to_seconds "$user_cpu")
  system_cpu_s=$(time_to_seconds "$system_cpu")

  # Convert memory to KB
  max_rss_kb=$(mem_to_kb "$max_rss")
  max_vmsize_kb=$(mem_to_kb "$max_vmsize")
  ave_rss_kb=$(mem_to_kb "$ave_rss")

  echo "${jid},${run_num},${elapsed_s},${total_cpu_s},${user_cpu_s},${system_cpu_s},${max_rss_kb},${max_vmsize_kb},${ave_rss_kb},${state},${exit_code}" >> "${CSV_FILE}"

  echo "  [Run ${run_num}] Job ${jid}: elapsed=${elapsed_s}s, cpu=${total_cpu_s}s, maxrss=${max_rss_kb}KB, state=${state}"
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Done! Results written to: ${CSV_FILE}"
echo ""
echo "Completed jobs:"
grep -c "COMPLETED" "${CSV_FILE}" || echo "0"
echo "Failed jobs:"
grep -c "FAILED\|CANCELLED\|TIMEOUT\|OUT_OF_MEMORY" "${CSV_FILE}" || echo "0"
echo ""
echo "To analyze, run:"
echo "  python3 ${SCRIPT_DIR}/bench_analyze.py ${CSV_FILE} <other_experiment.csv>"
echo "============================================================"
