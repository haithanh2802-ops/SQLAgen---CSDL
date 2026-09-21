from __future__ import annotations

import os
import re
from dataclasses import dataclass

MODEL_SIZE_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?:\b|[-_])", re.IGNORECASE)
GPU_HYBRID_MAX_BILLION = 4.0
KNOWN_MODEL_SIZES = {"qwen3.8": 27.0}


@dataclass(frozen=True)
class ModelRuntime:
    model_name: str
    size_billion: float | None
    num_gpu: int | None
    num_thread: int | None
    profile: str


def model_size_billions(model_name: str) -> float | None:
    match = MODEL_SIZE_PATTERN.search(model_name)
    if match:
        return float(match.group(1))
    family = model_name.partition(":")[0].casefold()
    return KNOWN_MODEL_SIZES.get(family)


def model_runtime(model_name: str, configured_num_gpu: int | None = None) -> ModelRuntime:
    """Choose GPU-first hybrid execution for small models and CPU for larger ones."""
    size = model_size_billions(model_name)
    cpu_threads = max(1, os.cpu_count() or 1)
    if configured_num_gpu is not None:
        profile = "explicit GPU override"
        num_gpu = configured_num_gpu
    elif size is not None and size <= GPU_HYBRID_MAX_BILLION:
        profile = "GPU-first with CPU fallback"
        # None lets Ollama choose GPU layers and spill to CPU when VRAM is limited.
        num_gpu = None
    else:
        profile = "CPU-focused"
        num_gpu = 0
    return ModelRuntime(
        model_name=model_name,
        size_billion=size,
        num_gpu=num_gpu,
        num_thread=cpu_threads,
        profile=profile,
    )
