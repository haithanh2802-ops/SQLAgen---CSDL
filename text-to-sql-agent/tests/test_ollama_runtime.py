import os

from text2sql_agent.ollama_runtime import model_runtime, model_size_billions


def test_model_size_parser_handles_small_and_large_tags():
    assert model_size_billions("qwen3.5:0.8b") == 0.8
    assert model_size_billions("qwen3.5:4b-instruct") == 4.0
    assert model_size_billions("qwen3-coder:30b") == 30.0
    assert model_size_billions("qwen3.8:latest") == 27.0


def test_small_models_use_gpu_first_with_cpu_threads():
    runtime = model_runtime("qwen3.5:4b")

    assert runtime.num_gpu is None
    assert runtime.num_thread == max(1, os.cpu_count() or 1)
    assert runtime.profile == "GPU-first with CPU fallback"


def test_large_or_unknown_models_use_cpu_focused_profile():
    assert model_runtime("qwen3:8b").num_gpu == 0
    assert model_runtime("qwen3-coder:30b").num_gpu == 0
    assert model_runtime("qwen3.8:latest").size_billion == 27.0
    assert model_runtime("qwen3.8:latest").profile == "CPU-focused"


def test_explicit_gpu_setting_overrides_model_profile():
    runtime = model_runtime("qwen3-coder:30b", configured_num_gpu=-1)

    assert runtime.num_gpu == -1
    assert runtime.profile == "explicit GPU override"
