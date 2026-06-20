from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class VLLMServerConfig:
    model: str
    served_model_name: str
    port: int
    kv_cache_dtype: str = "kvarn_k4v2_g128"
    block_size: int | None = None
    tokenizer: str | None = None
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_num_seqs: int = 64
    dtype: str = "float16"
    trust_remote_code: bool = True
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    reasoning_parser: str | None = None
    extra_args: tuple[str, ...] = ()
    extra_env: dict[str, str | int | float | bool | None] = field(default_factory=dict)


def block_size_from_kv_cache_dtype(kv_cache_dtype: str) -> int:
    if kv_cache_dtype.endswith("_g64"):
        return 64
    if kv_cache_dtype.endswith("_g128"):
        return 128
    return 128


def build_vllm_command(config: VLLMServerConfig) -> list[str]:
    block_size = config.block_size or block_size_from_kv_cache_dtype(
        config.kv_cache_dtype
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model,
        "--served-model-name",
        config.served_model_name,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--kv-cache-dtype",
        config.kv_cache_dtype,
        "--block-size",
        str(block_size),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]
    if config.tokenizer is not None:
        command += ["--tokenizer", config.tokenizer]
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    if config.enable_chunked_prefill:
        command.append("--enable-chunked-prefill")
    if config.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if config.reasoning_parser is not None:
        command += ["--reasoning-parser", config.reasoning_parser]
    command.extend(config.extra_args)
    return command


def build_vllm_env(config: VLLMServerConfig) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in config.extra_env.items():
        if value is not None:
            env[str(key)] = str(value)
    return env


def start_vllm_server(config: VLLMServerConfig) -> subprocess.Popen:
    command = build_vllm_command(config)
    print(f"[INFO] Starting vLLM server on port {config.port}")
    print("[INFO] Command:", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=build_vllm_env(config),
    )
    atexit.register(stop_vllm_server, process)
    return process


def wait_for_vllm_server(
    port: int,
    process: subprocess.Popen | None = None,
    timeout: int = 1200,
) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    start_time = time.time()
    while time.time() - start_time < timeout:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {process.returncode}."
            )
        try:
            urllib.request.urlopen(url, timeout=2)
            elapsed = time.time() - start_time
            print(f"[INFO] vLLM server ready on port {port} after {elapsed:.1f}s")
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM server did not start within {timeout}s.")


def stop_vllm_server(process: subprocess.Popen, timeout: int = 15) -> None:
    if process.poll() is not None:
        return

    print("[INFO] Shutting down vLLM server")
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
