"""GPU probe: decide whether the chosen Qwen model trains on this machine, from measurements.

Runs the real training setup (``train_qlora.build_trainer``) for a few complete
optimizer steps on the longest real training examples that fit the intended
sequence length, with the intended batch size and gradient accumulation, then
times an evaluation pass. Reports peak VRAM, seconds per optimizer step, tokens
per second, a projected training time, the adapter-health check (modules
attached, gradients, updates) and whether NVML energy telemetry works. Refuses to
run if another process holds GPU memory (stop the vLLM server first).

    uv run --extra gpu modelforge probe
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

DEFAULT_MODELS = ["Qwen/Qwen3-4B"]
PINNED_REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}
GATES = {"peak_reserved_gib_max": 10.5, "projected_training_hours_max": 4.0}
MIN_FREE_GIB = 10.5
_VISION = ("visual", "vision")


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


def discover_targets(linear_names: list[str], expected_layers: int | None) -> tuple[list[str] | str, dict]:
    """Choose LoRA targets from the names of a model's linear modules (pure; testable without torch).

    The text decoder is the ``<prefix>.layers.<n>.`` family with the most layers,
    skipping vision towers. Every linear module inside those layers is targeted.
    When the module-name suffixes are unique to the decoder (dense text models such
    as Qwen3), targets are returned as a plain suffix list, the form PEFT and vLLM's
    LoRA loader handle most reliably; otherwise (multimodal or hybrid checkpoints) an
    anchored regex keeps vision modules out.
    """
    pattern = re.compile(r"^(?P<prefix>.+?)\.layers\.(?P<idx>\d+)\.(?P<rest>.+)$")
    found = [(m["prefix"], int(m["idx"]), m["rest"]) for name in linear_names
             if not any(v in name for v in _VISION) and (m := pattern.match(name))]
    layers_by_prefix = Counter()
    for prefix in {p for p, _, _ in found}:
        layers_by_prefix[prefix] = len({i for p, i, _ in found if p == prefix})
    if not layers_by_prefix:
        return [], {"expected_layers": expected_layers, "coverage_ok": False, "reason": "no decoder layers found"}
    prefix = layers_by_prefix.most_common(1)[0][0]
    per_layer: dict[int, set[str]] = {}
    for p, idx, rest in found:
        if p == prefix:
            per_layer.setdefault(idx, set()).add(rest)
    suffixes = sorted({rest.rsplit(".", 1)[-1] for modules in per_layer.values() for rest in modules})
    outside = [n for n in linear_names if not n.startswith(prefix + ".layers.") and n.rsplit(".", 1)[-1] in suffixes]
    if outside:
        rests = sorted({rest for modules in per_layer.values() for rest in modules})
        targets: list[str] | str = (r"^" + re.escape(prefix) + r"\.layers\.\d+\.(?:"
                                    + "|".join(re.escape(r) for r in rests) + r")$")
    else:
        targets = suffixes
    missing = [layer for layer in range(expected_layers or 0)
               if not any("attn" in m or "attention" in m for m in per_layer.get(layer, set()))
               or not any("mlp" in m for m in per_layer.get(layer, set()))]
    report = {"decoder_prefix": prefix, "expected_layers": expected_layers, "layers_with_linear": len(per_layer),
              "suffixes": suffixes, "target_form": "regex" if outside else "suffix_list",
              "layers_missing_attention_or_mlp": missing,
              "coverage_ok": bool(expected_layers) and not missing and len(per_layer) == expected_layers,
              "example_layer_modules": {k: sorted(per_layer[k]) for k in sorted(per_layer)[:2]}}
    return targets, report


def language_lora_targets(model) -> tuple[list[str] | str, dict]:
    import torch.nn as nn
    linear_types: tuple = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_types = (nn.Linear, bnb.nn.Linear4bit)
    except ImportError:
        pass
    names = [name for name, module in model.named_modules() if isinstance(module, linear_types)]
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    return discover_targets(names, getattr(text_config, "num_hidden_layers", None))


def _telemetry() -> dict:
    from modelforge.energy import GpuEnergyMeter

    with GpuEnergyMeter(sample_seconds=0.2) as meter:
        time.sleep(2.0)
    reading = meter.reading()
    return {"method": reading.method, "idle_joules_2s": reading.gpu_energy_joules, "notes": reading.notes,
            "scope": "GPU only"}


def _probe_training(model_id: str, data_dir: Path, steps: int, grad_accum: int, max_seq_len: int,
                    eval_examples: int, epochs: float) -> dict:
    import torch
    from transformers import AutoTokenizer

    from modelforge.energy import GpuEnergyMeter
    from modelforge.training.callbacks import AdapterHealthCallback, StepTimer
    from modelforge.training.train_qlora import build_trainer, load_splits

    revision = PINNED_REVISIONS[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    splits = load_splits(tokenizer, data_dir, max_seq_len, enable_thinking=False)
    train_rows, train_info = splits["train"]
    val_rows, val_info = splits["val"]
    longest = sorted(train_rows, key=lambda r: r["n_tokens"], reverse=True)[: steps * grad_accum]
    eval_subset = sorted(val_rows, key=lambda r: r["n_tokens"], reverse=True)[:eval_examples]
    args = Namespace(model=model_id, revision=revision, out=Path("artifacts/probe/run"), max_seq_len=max_seq_len,
                     epochs=1.0, lr=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05, batch_size=1,
                     grad_accum=grad_accum, seed=42, max_steps=steps, save_steps=0, eval_steps=0,
                     save_total_limit=0, logging_steps=1)
    health = AdapterHealthCallback(check_after=min(3, steps), fail_fast=False)
    timer = StepTimer()
    torch.cuda.reset_peak_memory_stats()
    trainer, info = build_trainer(args, tokenizer, longest, eval_subset, callbacks=[health, timer],
                                  output_dir=Path("artifacts/probe/run"))
    weights_gib = round(torch.cuda.memory_allocated() / 2**30, 3)
    with GpuEnergyMeter() as meter:
        trainer.train()
    reading = meter.reading()
    train_peak = {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                  "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
    tokens_per_step = sum(r["n_tokens"] for r in longest) / steps
    steady = timer.step_seconds[1:] or timer.step_seconds
    seconds_per_step = sum(steady) / len(steady)
    tokens_per_second = tokens_per_step / seconds_per_step
    projected_hours = train_info["kept_tokens"] * epochs / tokens_per_second / 3600

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    metrics = trainer.evaluate()
    eval_seconds = time.perf_counter() - t0
    per_example = eval_seconds / max(len(eval_subset), 1)
    result = {
        "model": model_id, "revision": revision, "model_class": info["model_class"],
        "lora_targets": info["lora_targets"], "coverage": info["coverage"], "adapter_health": health.report,
        "weights_gib_after_load": weights_gib, "train_peak_vram_gib": train_peak,
        "eval_peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "config": {"steps": steps, "grad_accum": grad_accum, "batch_size": 1, "max_seq_len": max_seq_len,
                   "examples": "longest training examples that fit max_seq_len", "examples_used": len(longest)},
        "seconds_per_optimizer_step": [round(s, 2) for s in timer.step_seconds],
        "tokens_per_second_steady": round(tokens_per_second, 1),
        "projection": {"epochs": epochs, "kept_train_examples": train_info["kept"],
                       "kept_train_tokens": train_info["kept_tokens"], "projected_hours": round(projected_hours, 2),
                       "note": "from the longest examples, so per-token cost is at the pessimistic end"},
        "eval": {"examples": len(eval_subset), "seconds": round(eval_seconds, 1), "eval_loss": metrics.get("eval_loss"),
                 "seconds_per_example": round(per_example, 3),
                 "projected_full_val_seconds": round(per_example * val_info["kept"], 1),
                 "full_val_kept_examples": val_info["kept"]},
        "dataset_lengths": {"train": train_info, "val": val_info},
        "gpu_energy_probe_steps": {"method": reading.method, "joules": reading.gpu_energy_joules, "notes": reading.notes},
    }
    result["gates"] = {
        "adapter_health_ok": health.report.get("status") == "ok",
        "coverage_ok": bool(info["coverage"].get("coverage_ok")),
        "peak_reserved_within_limit": train_peak["reserved"] <= GATES["peak_reserved_gib_max"],
        "projected_training_within_limit": projected_hours <= GATES["projected_training_hours_max"],
    }
    result["gates"]["all_passed"] = all(result["gates"].values())
    del trainer
    torch.cuda.empty_cache()
    return result


def main() -> int:
    from modelforge.envinfo import environment

    report: dict = {"environment": environment(), "disk_free_gib": round(shutil.disk_usage(Path.home()).free / 2**30, 1),
                    "gates": GATES}
    import torch
    if not torch.cuda.is_available():
        print(json.dumps(report, indent=1, default=str))
        print("CUDA is not available; stopping.")
        return 1
    free, total = torch.cuda.mem_get_info()
    report["gpu_memory_before_gib"] = {"free": round(free / 2**30, 2), "total": round(total / 2**30, 2)}
    if free / 2**30 < MIN_FREE_GIB:
        print(json.dumps(report, indent=1, default=str))
        print(f"Only {free / 2**30:.1f} GiB of GPU memory is free; stop the vLLM server (or other GPU work) first.")
        return 1
    report["telemetry"] = _telemetry()
    data_dir = Path(os.environ.get("PROBE_DATA", "artifacts/training/finqa-grouped-v1"))
    report["models"] = []
    for model_id in os.environ.get("PROBE_MODELS", ",".join(DEFAULT_MODELS)).split(","):
        try:
            report["models"].append(_probe_training(
                model_id.strip(), data_dir, steps=int(os.environ.get("PROBE_STEPS", "3")),
                grad_accum=int(os.environ.get("PROBE_GRAD_ACCUM", "16")),
                max_seq_len=int(os.environ.get("PROBE_MAX_SEQ_LEN", "3072")),
                eval_examples=int(os.environ.get("PROBE_EVAL_EXAMPLES", "32")),
                epochs=float(os.environ.get("PROBE_EPOCHS", "2"))))
        except Exception as exc:  # noqa: BLE001 - the probe reports, it does not crash
            report["models"].append({"model": model_id, "error": f"{type(exc).__name__}: {exc}"})
    out = Path("artifacts/probe")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))
    print(f"\nReport written to {path}")
    return 0
