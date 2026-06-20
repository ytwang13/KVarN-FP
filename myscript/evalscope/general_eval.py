from __future__ import annotations

import copy
from typing import Any

from evalscope import TaskConfig, run_task


GENERAL_DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "arc": {"args": {"few_shot_num": 0}},
    "piqa": {"args": {"few_shot_num": 0}},
    "hellaswag": {"args": {"few_shot_num": 0}},
    "mmlu": {"args": {"few_shot_num": 0}},
    "winogrande": {"args": {"few_shot_num": 0}},
}


def split_tasks(tasks: str | list[str]) -> list[str]:
    if isinstance(tasks, str):
        return [task.strip() for task in tasks.split(",") if task.strip()]
    return list(tasks)


def per_dataset_limit(limit: int | float, tasks: list[str]) -> int | float:
    if limit != -1 and tasks and isinstance(limit, int) and not isinstance(limit, bool):
        return max(1, limit // len(tasks))
    return limit


def evaluate_general(
    *,
    model: str,
    api_url: str,
    tasks: str | list[str],
    generation_config: dict[str, Any],
    limit: int | float = -1,
    eval_batch_size: int = 32,
    repeats: int = 1,
    seed: int = 42,
    timeout: int = 60000,
    stream: bool = True,
    dataset_dir: str | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Run evalscope general-task evaluation through a vLLM OpenAI API."""
    task_list = split_tasks(tasks)
    valid_tasks = [task for task in task_list if task in GENERAL_DATASET_CONFIGS]
    task_limit = per_dataset_limit(limit, valid_tasks)

    all_results: dict[str, Any] = {}
    for dataset_name in task_list:
        if dataset_name not in GENERAL_DATASET_CONFIGS:
            print(f"[WARN] Unknown general dataset '{dataset_name}', skipping.")
            continue

        print(f"[INFO] Running general dataset={dataset_name}, limit={task_limit}")
        gen_config = copy.deepcopy(generation_config)
        gen_config.setdefault("timeout", timeout)
        gen_config.setdefault("stream", stream)
        task_kwargs: dict[str, Any] = {
            "model": model,
            "api_url": api_url,
            "eval_type": "openai_api",
            "datasets": [dataset_name],
            "dataset_args": {
                dataset_name: dict(GENERAL_DATASET_CONFIGS[dataset_name]["args"])
            },
            "generation_config": gen_config,
            "eval_batch_size": eval_batch_size,
            "repeats": repeats,
            "seed": seed,
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

    print("[SUMMARY] General Evaluation Results:")
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
