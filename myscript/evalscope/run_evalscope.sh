#!/usr/bin/env bash
# Run KVarN evalscope tasks inside the Singularity/conda environment.
#
# Expected compute-node setup:
#   cd /scratch/yw6594/quant/KVarN
#   singularity exec --nv --fakeroot --overlay /scratch/yw6594/sig/overlay-50G-10M-wsvd.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
#   source /ext3/env.sh
#   conda activate kvarn
#   bash myscript/evalscope/run_evalscope.sh smoke

set -euo pipefail

ROOT="${KVAR_N_ROOT:-/scratch/yw6594/quant/KVarN}"
cd "$ROOT"

TASK_TYPE="${1:-smoke}"
MODEL="${EVALSCOPE_MODEL:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${EVALSCOPE_SERVED_MODEL_NAME:-kvarn}"
KV_CACHE_DTYPE="${EVALSCOPE_KV_CACHE_DTYPE:-kvarn_k4v2_g128}"
MAX_MODEL_LEN="${EVALSCOPE_MAX_MODEL_LEN:-4096}"
TP_SIZE="${EVALSCOPE_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${EVALSCOPE_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${EVALSCOPE_MAX_NUM_SEQS:-8}"
EVAL_BATCH_SIZE="${EVALSCOPE_BATCH_SIZE:-8}"
PORT="${EVALSCOPE_PORT:-8801}"
LIMIT="${EVALSCOPE_LIMIT:--1}"
SMOKE_LIMIT="${EVALSCOPE_SMOKE_LIMIT:-2}"
GENERAL_TASKS="${EVALSCOPE_GENERAL_TASKS:-arc,piqa,hellaswag,mmlu,winogrande}"
REASONING_TASKS="${EVALSCOPE_REASONING_TASKS:-gsm8k,math_500,aime24,aime25,aime26}"
DATASET_DIR="${EVALSCOPE_DATASET_DIR:-}"
OUTPUT_ROOT="${EVALSCOPE_OUTPUT_ROOT:-myscript/output/evalscope}"
TIMESTAMP="${EVALSCOPE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

case "$KV_CACHE_DTYPE" in
  *_g64) BLOCK_SIZE="${EVALSCOPE_BLOCK_SIZE:-64}" ;;
  *) BLOCK_SIZE="${EVALSCOPE_BLOCK_SIZE:-128}" ;;
esac

sanitize_name() {
  printf '%s' "$1" | tr '/: ,' '____' | tr -cd 'A-Za-z0-9_.-'
}

kv_label="$(printf '%s' "$KV_CACHE_DTYPE" | sed -E 's/_g[0-9]+$//')"
case "$TASK_TYPE" in
  reasoning) task_label="reason" ;;
  *) task_label="$TASK_TYPE" ;;
esac

run_name="$(sanitize_name "${EVALSCOPE_RUN_NAME:-${kv_label}_${task_label}}")"
if [ "${EVALSCOPE_ENABLE_THINKING:-0}" = "1" ]; then
  run_name="$(sanitize_name "${run_name}_think")"
fi

task_dir="$(sanitize_name "$TASK_TYPE")"
if [ -n "${EVALSCOPE_WORK_DIR:-}" ]; then
  WORK_DIR="$EVALSCOPE_WORK_DIR"
else
  WORK_DIR="${OUTPUT_ROOT}/${run_name}/${task_dir}/${TIMESTAMP}"
fi

cmd=(
  python myscript/evalscope/run_evaluation.py
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --task-types "$TASK_TYPE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --block-size "$BLOCK_SIZE"
  --vllm-max-model-len "$MAX_MODEL_LEN"
  --vllm-tensor-parallel-size "$TP_SIZE"
  --vllm-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --vllm-max-num-seqs "$MAX_NUM_SEQS"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --port "$PORT"
  --limit "$LIMIT"
  --smoke-limit "$SMOKE_LIMIT"
  --general-tasks "$GENERAL_TASKS"
  --reasoning-tasks "$REASONING_TASKS"
  --work-dir "$WORK_DIR"
)

if [ -n "$DATASET_DIR" ]; then
  cmd+=(--dataset-dir "$DATASET_DIR")
fi

if [ "${EVALSCOPE_ENABLE_THINKING:-0}" = "1" ]; then
  cmd+=(--enable-thinking)
fi

if [ "${EVALSCOPE_USE_EXISTING_SERVER:-0}" = "1" ]; then
  cmd+=(--use-existing-server)
fi

echo "[INFO] $(date)"
echo "[INFO] Output directory: ${WORK_DIR}"
echo "[INFO] Running: ${cmd[*]}"
"${cmd[@]}"
