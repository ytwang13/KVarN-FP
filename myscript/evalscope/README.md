# KVarN evalscope evaluation

This directory runs evalscope benchmarks against KVarN through vLLM's
OpenAI-compatible server. KVarN is selected with normal vLLM flags, so there is
no extra model integration layer here.

## Enter the environment

On a compute node:

```bash
cd /scratch/yw6594/quant/KVarN
singularity exec --nv --fakeroot --overlay /scratch/yw6594/sig/overlay-50G-10M-wsvd.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
source /ext3/env.sh
conda activate kvarn
```

## Smoke test

```bash
bash myscript/evalscope/run_evalscope.sh smoke
```

The smoke run evaluates two ARC samples. Override defaults with environment
variables:

```bash
EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_SMOKE_LIMIT=2 \
EVALSCOPE_MAX_MODEL_LEN=4096 \
EVALSCOPE_BATCH_SIZE=2 \
bash myscript/evalscope/run_evalscope.sh smoke
```

## General tasks

Default task set: `arc,piqa,hellaswag,mmlu,winogrande`.

```bash
EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_MAX_MODEL_LEN=32768 \
EVALSCOPE_BATCH_SIZE=32 \
bash myscript/evalscope/run_evalscope.sh general
```

Small sanity pass:

```bash
EVALSCOPE_LIMIT=50 bash myscript/evalscope/run_evalscope.sh general
```

## Reasoning tasks

Default task set: `gsm8k,math_500,aime24,aime25`.

```bash
EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_MAX_MODEL_LEN=32768 \
EVALSCOPE_BATCH_SIZE=32 \
EVALSCOPE_ENABLE_THINKING=1 \
bash myscript/evalscope/run_evalscope.sh reasoning
```

This writes to a timestamped directory like:

```bash
myscript/output/evalscope/kvarn_k4v2_reason_think/reasoning/20260617_052000
```

```bash

EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_MAX_MODEL_LEN=32768 \
EVALSCOPE_BATCH_SIZE=32 \
EVALSCOPE_ENABLE_THINKING=1 \
EVALSCOPE_REASONING_TASKS=aime24,aime25 \
nohup bash myscript/evalscope/run_evalscope.sh reasoning > myscript/logs/eval_kvarnk4v2_evalscope_reasoning.nohup.log 2>&1 &
# makeup aime24 25 for that
tail -f myscript/logs/eval_kvarnk4v2_evalscope_reasoning.nohup.log


EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_MAX_MODEL_LEN=32768 \
EVALSCOPE_KV_CACHE_DTYPE=kvarn_k2v2_g128 \
EVALSCOPE_BATCH_SIZE=32 \
EVALSCOPE_ENABLE_THINKING=1 \
bash myscript/evalscope/run_evalscope.sh reasoning

EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_MAX_MODEL_LEN=32768 \
EVALSCOPE_KV_CACHE_DTYPE=kvarn_k2v2_g128 \
EVALSCOPE_BATCH_SIZE=32 \
EVALSCOPE_ENABLE_THINKING=1 \
bash myscript/evalscope/run_evalscope.sh general

EVALSCOPE_MODEL=Qwen/Qwen3-4B \
EVALSCOPE_MAX_MODEL_LEN=32768 \
EVALSCOPE_BATCH_SIZE=32 \
EVALSCOPE_ENABLE_THINKING=1 \
bash myscript/evalscope/run_evalscope.sh general
```

## Existing server

If vLLM is already running on port 8801:

```bash
EVALSCOPE_USE_EXISTING_SERVER=1 \
bash myscript/evalscope/run_evalscope.sh smoke
```

## Useful variables

- `EVALSCOPE_MODEL`: model id/path, default `Qwen/Qwen3-4B`
- `EVALSCOPE_KV_CACHE_DTYPE`: default `kvarn_k4v2_g128`
- `EVALSCOPE_BLOCK_SIZE`: inferred as `128` for `_g128`, `64` for `_g64`
- `EVALSCOPE_LIMIT`: evalscope sample limit for general/reasoning
- `EVALSCOPE_DATASET_DIR`: evalscope dataset cache directory
- `EVALSCOPE_OUTPUT_ROOT`: output root, default `myscript/output/evalscope`
- `EVALSCOPE_RUN_NAME`: run group name; default is derived from KV/task/thinking
- `EVALSCOPE_TIMESTAMP`: timestamp folder name, default `YYYYmmdd_HHMMSS`
- `EVALSCOPE_WORK_DIR`: exact output directory override
- `EVALSCOPE_USE_EXISTING_SERVER`: set `1` to skip launching vLLM
