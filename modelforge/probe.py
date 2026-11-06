"""GPU probe: decide the Qwen size and serving backend from measurements, not guesses.

Checks the driver, CUDA, free VRAM, NVML energy telemetry, 4-bit loading, and the
peak memory of a QLoRA training step at several sequence lengths. Writes a JSON
report to ``artifacts/probe/``. Run on the WSL2 GPU machine:

    uv run --extra gpu modelforge probe            # Qwen3.5-4B
    PROBE_MODELS=Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B uv run --extra gpu modelforge probe
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT_MODELS = ["Qwen/Qwen3.5-4B"]
PINNED_REVISIONS = {"Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
SEQ_LENGTHS = [1024, 2048, 3072, 4096]


def load_text_model(model_id: str, revision: str | None, quantization):
    """Load the text path of a (possibly multimodal) Qwen checkpoint in 4-bit."""
    import torch
    from transformers import AutoModelForCausalLM

    errors = []
    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    for loader in loaders:
        try:
            model = loader.from_pretrained(model_id, revision=revision, quantization_config=quantization,
                                           dtype=torch.bfloat16, device_map={"": 0})
            return model, f"{loader.__name__}->{type(model).__name__}"
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError("could not load model: " + " | ".join(errors))


def language_lora_targets(model) -> tuple[str, dict]:
    """Discover every linear layer in the language model's decoder layers.

    Qwen3.5 is hybrid (linear-attention and full-attention layers use different
    projection names), so fixed name lists silently miss most layers. Returns a
    PEFT target regex and a coverage report; ``coverage_ok`` requires every decoder
    layer to have at least one adapted attention/mixer module and one MLP module.
    """
    import re
    from collections import defaultdict

    import torch.nn as nn
    linear_types: tuple = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_types = (nn.Linear, bnb.nn.Linear4bit)
    except ImportError:
        pass
    per_layer: dict[int, set[str]] = defaultdict(set)
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if "language_model" not in name or "visual" in name or not isinstance(module, linear_types):
            continue
        match = re.search(r"\.layers\.(\d+)\.(.+)$", name)
        if not match:
            continue
        per_layer[int(match.group(1))].add(match.group(2))
        suffixes.add(name.rsplit(".", 1)[-1])
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    expected = getattr(text_config, "num_hidden_layers", None)
    missing = []
    for layer in range(expected or 0):
        modules = per_layer.get(layer, set())
        if not any("attn" in m or "attention" in m for m in modules) or not any("mlp" in m for m in modules):
            missing.append(layer)
    regex = r".*language_model.*\.layers\.\d+\..*\.(" + "|".join(sorted(re.escape(x) for x in suffixes)) + r")$"
    report = {"expected_layers": expected, "layers_with_linear": len(per_layer), "suffixes": sorted(suffixes),
              "layers_missing_attention_or_mlp": missing,
              "coverage_ok": bool(expected) and not missing and len(per_layer) == expected,
              "example_layer_modules": {k: sorted(per_layer[k]) for k in sorted(per_layer)[:4]}}
    return regex, report


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def _telemetry() -> dict:
    from modelforge.energy import GpuEnergyMeter

    with GpuEnergyMeter(sample_seconds=0.2) as meter:
        time.sleep(1.5)
    reading = meter.reading()
    return {"method": reading.method, "joules_over_idle_1.5s": reading.gpu_energy_joules, "notes": reading.notes}


def _training_step_memory(model_id: str, revision: str | None) -> dict:
    import torch
    from huggingface_hub import model_info
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    revision = revision or PINNED_REVISIONS.get(model_id) or model_info(model_id).sha
    result: dict = {"model": model_id, "revision": revision}
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    model, model_class = load_text_model(model_id, revision, quantization)
    result["model_class"] = model_class
    result["weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
    targets, coverage = language_lora_targets(model)
    result["lora_targets"] = {"regex": targets, **coverage}
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["lora_modules_matched"] = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    if not coverage["coverage_ok"]:
        result["WARNING"] = "LoRA target discovery did not cover every decoder layer; training will refuse to run"
    steps = {}
    for length in SEQ_LENGTHS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(100, 1000, (1, length), device="cuda")
        try:
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            steps[length] = {"ok": True, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        except torch.cuda.OutOfMemoryError:
            steps[length] = {"ok": False, "error": "OOM"}
            break
    result["qlora_step_by_seq_len"] = steps
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    report: dict = {"host": platform.node(), "platform": platform.platform(),
                    "wsl": "microsoft" in Path("/proc/version").read_text().lower() if Path("/proc/version").exists() else False,
                    "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,power.limit",
                                        "--format=csv,noheader"]),
                    "disk_free_gib": round(shutil.disk_usage(Path.home()).free / 2**30, 1)}
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            report["torch"].update(device=torch.cuda.get_device_name(0), free_gib=round(free / 2**30, 2),
                                   total_gib=round(total / 2**30, 2), bf16=torch.cuda.is_bf16_supported())
    except ImportError as exc:
        report["torch"] = f"unavailable: {exc}"
    report["telemetry"] = _telemetry()
    try:
        import vllm
        report["vllm"] = vllm.__version__
    except ImportError as exc:
        report["vllm"] = f"unavailable: {exc}"
    report["models"] = []
    for model_id in os.environ.get("PROBE_MODELS", ",".join(DEFAULT_MODELS)).split(","):
        try:
            report["models"].append(_training_step_memory(model_id.strip(), None))
        except Exception as exc:  # noqa: BLE001 - the probe reports, it does not crash
            report["models"].append({"model": model_id, "error": f"{type(exc).__name__}: {exc}"})
    out = Path("artifacts/probe")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))
    print(f"\nReport written to {path}")
    return 0
