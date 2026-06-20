from __future__ import annotations

import copy
from typing import Any

from evalscope import TaskConfig, run_task
from general_eval import per_dataset_limit, split_tasks


REASONING_DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "gsm8k": {"args": {"few_shot_num": 4}, "gen": "mcq"},
    "humaneval": {
        "args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"},
        "gen": "mcq",
    },
    "live_code_bench": {"args": {"few_shot_num": 0}, "gen": "mcq"},
    "gpqa_diamond": {
        "args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"},
        "gen": "mcq",
    },
    "math_500": {
        "args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"},
        "gen": "math",
    },
    "aime24": {
        "args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"},
        "gen": "aime",
    },
    "aime25": {
        "args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"},
        "gen": "aime",
    },
    "aime26": {
        "args": {"few_shot_num": 0, "aggregation": "mean_and_pass_at_k"},
        "gen": "aime",
    },
}
AIME_REPEATS = 8


def evaluate_reasoning(
    *,
    model: str,
    api_url: str,
    tasks: str | list[str],
    generation_configs: dict[str, dict[str, Any]],
    limit: int | float = -1,
    eval_batch_size: int = 32,
    repeats: int = 1,
    seed: int = 42,
    timeout: int = 60000,
    stream: bool = True,
    dataset_dir: str | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Run evalscope reasoning-task evaluation through a vLLM OpenAI API."""
    task_list = split_tasks(tasks)
    valid_tasks = [task for task in task_list if task in REASONING_DATASET_CONFIGS]
    task_limit = per_dataset_limit(limit, valid_tasks)

    all_results: dict[str, Any] = {}
    for dataset_name in task_list:
        if dataset_name not in REASONING_DATASET_CONFIGS:
            print(f"[WARN] Unknown reasoning dataset '{dataset_name}', skipping.")
            continue

        entry = REASONING_DATASET_CONFIGS[dataset_name]
        gen_type = entry["gen"]
        if gen_type not in generation_configs:
            print(
                f"[WARN] No generation config for '{gen_type}' "
                f"(dataset '{dataset_name}'), skipping."
            )
            continue

        task_repeats = AIME_REPEATS if dataset_name.startswith("aime") else repeats
        print(
            f"[INFO] Running reasoning dataset={dataset_name}, "
            f"repeats={task_repeats}, limit={task_limit}"
        )
        task_kwargs: dict[str, Any] = {
            "model": model,
            "api_url": api_url,
            "eval_type": "openai_api",
            "datasets": [dataset_name],
            "dataset_args": {dataset_name: dict(entry["args"])},
            "generation_config": copy.deepcopy(generation_configs[gen_type]),
            "eval_batch_size": eval_batch_size,
            "repeats": task_repeats,
            "seed": seed,
            "timeout": timeout,
            "stream": stream,
        }
        if limit != -1:
            task_kwargs["limit"] = task_limit
        if dataset_dir is not None:
            task_kwargs["dataset_dir"] = dataset_dir
        if work_dir is not None:
            task_kwargs["work_dir"] = work_dir

        result = run_task(task_cfg=TaskConfig(**task_kwargs))
        all_results[dataset_name] = result
        _print_scores(dataset_name, result)

    print("[SUMMARY] Reasoning Evaluation Results:")
    for dataset_name, result in all_results.items():
        _print_scores(dataset_name, result, prefix="  ")
    return all_results


def _print_scores(dataset_name: str, result: Any, prefix: str = "[SCORE] ") -> None:
    if isinstance(result, dict):
        for benchmark_name, report in result.items():
            score = getattr(report, "score", "N/A")
            print(f"{prefix}{dataset_name} ({benchmark_name}): {score}")
    else:
        print(f"{prefix}{dataset_name}: {result}")
