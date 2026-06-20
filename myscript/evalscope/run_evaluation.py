from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_eval import evaluate_general
from reasoning_eval import evaluate_reasoning
from vllm_server import VLLMServerConfig, start_vllm_server, wait_for_vllm_server


DEFAULT_GENERAL_TASKS = "arc,piqa,hellaswag,mmlu,winogrande"
DEFAULT_REASONING_TASKS = (
    "gsm8k,humaneval,live_code_bench,gpqa_diamond,math_500,aime24,aime25,aime26"
)
TASK_TYPES = ("smoke", "general", "reasoning")


def parse_limit(value: str) -> int | float:
    if value == "-1":
        return -1
    if "." in value:
        return float(value)
    return int(value)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_task_types(value: str, parser: argparse.ArgumentParser) -> list[str]:
    selected: list[str] = []
    for task_type in split_csv(value):
        if task_type not in TASK_TYPES:
            parser.error(f"Unknown task type: {task_type}")
        if task_type not in selected:
            selected.append(task_type)
    if not selected:
        parser.error("--task-types cannot be empty.")
    return selected


def generation_config(max_tokens: int, enable_thinking: bool) -> dict[str, Any]:
    if enable_thinking:
        temperature = 0.6
        top_p = 0.95
    else:
        temperature = 0.7
        top_p = 0.8
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 20,
        "n": 1,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    }


def reasoning_generation_configs(enable_thinking: bool) -> dict[str, dict[str, Any]]:
    return {
        "mcq": generation_config(28672, enable_thinking),
        "math": generation_config(30720, enable_thinking),
        "aime": generation_config(30720, enable_thinking),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KVarN evalscope tasks via vLLM.")
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument(
        "--task-types",
        default="smoke",
        help="Comma-separated task families: smoke,general,reasoning.",
    )
    parser.add_argument("--general-tasks", default=DEFAULT_GENERAL_TASKS)
    parser.add_argument("--reasoning-tasks", default=DEFAULT_REASONING_TASKS)
    parser.add_argument("--limit", type=parse_limit, default=-1)
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--work-dir", default="myscript/output/evalscope")
    parser.add_argument("--request-timeout", type=int, default=60000)
    parser.add_argument("--served-model-name", default="kvarn")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--use-existing-server", action="store_true")
    parser.add_argument("--vllm-startup-timeout", type=int, default=1200)
    parser.add_argument("--kv-cache-dtype", default="kvarn_k4v2_g128")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--reasoning-parser", default=None)
    parser.add_argument("--general-max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--vllm-extra-arg",
        action="append",
        default=[],
        help="Additional raw argument passed to vLLM. Repeat for each token.",
    )
    args = parser.parse_args()
    args.selected_task_types = parse_task_types(args.task_types, parser)
    return args


def main() -> None:
    args = parse_args()
    prepare_work_dir(args.work_dir)
    server_process = None
    try:
        if not args.use_existing_server:
            config = VLLMServerConfig(
                model=args.model,
                served_model_name=args.served_model_name,
                port=args.port,
                kv_cache_dtype=args.kv_cache_dtype,
                block_size=args.block_size,
                tokenizer=args.tokenizer,
                max_model_len=args.vllm_max_model_len,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                max_num_seqs=args.vllm_max_num_seqs,
                dtype=args.dtype,
                reasoning_parser=args.reasoning_parser,
                extra_args=tuple(args.vllm_extra_arg),
            )
            server_process = start_vllm_server(config)
            wait_for_vllm_server(
                args.port,
                server_process,
                timeout=args.vllm_startup_timeout,
            )
        elif args.api_url is None:
            wait_for_vllm_server(args.port, timeout=args.vllm_startup_timeout)

        if args.server_only:
            if server_process is None:
                print("[INFO] Existing vLLM server is ready.")
                return
            print(f"[INFO] vLLM server is running on port {args.port}.")
            server_process.wait()
            return

        api_url = args.api_url or f"http://127.0.0.1:{args.port}/v1/chat/completions"
        results: dict[str, Any] = {}
        if "smoke" in args.selected_task_types:
            results["smoke"] = evaluate_general(
                model=args.served_model_name,
                api_url=api_url,
                tasks="arc",
                generation_config=generation_config(args.general_max_tokens, False),
                limit=args.smoke_limit,
                eval_batch_size=min(args.eval_batch_size, args.smoke_limit),
                seed=args.seed,
                timeout=args.request_timeout,
                dataset_dir=args.dataset_dir,
                work_dir=args.work_dir,
            )
        if "general" in args.selected_task_types:
            results["general"] = evaluate_general(
                model=args.served_model_name,
                api_url=api_url,
                tasks=args.general_tasks,
                generation_config=generation_config(args.general_max_tokens, False),
                limit=args.limit,
                eval_batch_size=args.eval_batch_size,
                seed=args.seed,
                timeout=args.request_timeout,
                dataset_dir=args.dataset_dir,
                work_dir=args.work_dir,
            )
        if "reasoning" in args.selected_task_types:
            results["reasoning"] = evaluate_reasoning(
                model=args.served_model_name,
                api_url=api_url,
                tasks=args.reasoning_tasks,
                generation_configs=reasoning_generation_configs(args.enable_thinking),
                limit=args.limit,
                eval_batch_size=args.eval_batch_size,
                seed=args.seed,
                timeout=args.request_timeout,
                dataset_dir=args.dataset_dir,
                work_dir=args.work_dir,
            )
        print("[RESULT]", results)
    finally:
        if server_process is not None:
            from vllm_server import stop_vllm_server

            stop_vllm_server(server_process)


def prepare_work_dir(work_dir: str) -> None:
    path = Path(work_dir)
    blocked = next((parent for parent in (path, *path.parents) if parent.is_file()), None)
    if blocked is not None:
        raise NotADirectoryError(
            f"Evalscope work directory '{work_dir}' cannot be created because "
            f"'{blocked}' is a file. Set --work-dir or EVALSCOPE_WORK_DIR to "
            "a directory path."
        )
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    main()
