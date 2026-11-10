"""Start vLLM for the Qwen comparison with fixed, recorded settings (run on the GPU machine).

The server listens on 127.0.0.1 only (reach it through an SSH tunnel). LoRA is
always enabled, so the unchanged and fine-tuned models are served by the same
server configuration: requests choose the base (its model id) or an adapter (its
registered name). The model's generation_config.json is ignored
(``--generation-config vllm``) and every request sends its sampling parameters.
A serving record (command, environment, tokenizer fingerprint, adapter hashes) is
written before vLLM starts.

    uv run --extra gpu modelforge serve --model Qwen/Qwen3-4B --revision <sha> \
        [--lora qwen3-4b-finqa-v1=artifacts/adapters/qwen3-4b-finqa-v1/adapter]
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


# FlashInfer's top-k/top-p sampler needs prebuilt kernels or nvcc, neither of which is installed on the
# WSL box; vLLM then fails at its startup sampling pass. Greedy decoding never uses top-k/top-p, and the
# base and tuned models share the server, so the PyTorch sampler changes nothing in the comparison.
SERVE_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def build_command(model: str, revision: str, *, port: int = 8000, max_model_len: int = 5120,
                  gpu_memory_utilization: float = 0.87, max_lora_rank: int = 16,
                  loras: dict[str, Path] | None = None, compact_json: bool = False,
                  reasoning_parser: str | None = None, lora: bool = True, language_model_only: bool = False,
                  limit_mm_per_prompt: str | None = None, cpu_offload_gb: float | None = None,
                  enforce_eager: bool = False) -> list[str]:
    executable = shutil.which("vllm")
    base = [executable] if executable else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    command = base + [
        "serve", model, "--revision", revision, "--tokenizer-revision", revision,
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--seed", "0", "--generation-config", "vllm",
    ]
    if lora:
        command += ["--enable-lora", "--max-lora-rank", str(max_lora_rank), "--max-loras", "1"]
    elif loras:
        raise ValueError("adapters need LoRA enabled")
    if language_model_only:   # multimodal checkpoints (e.g. Gemma 4): load only the text model
        command += ["--language-model-only"]
    if limit_mm_per_prompt:
        command += ["--limit-mm-per-prompt", limit_mm_per_prompt]
    if cpu_offload_gb:
        command += ["--cpu-offload-gb", str(cpu_offload_gb)]
    if enforce_eager:     # no torch.compile / CUDA graphs (vLLM 0.33 cannot compile Gemma 4 with offloaded weights)
        command += ["--enforce-eager"]
    if compact_json:
        # Server-wide in vLLM 0.33 (the per-request field is ignored): guided JSON without free whitespace, so a
        # model cannot emit whitespace until max_tokens. Used for FinSight serving; FinQA runs keep the default
        # protocol they were measured with.
        command += ["--structured-outputs-config", '{"backend": "xgrammar", "disable_any_whitespace": true}']
    if reasoning_parser:  # thinking requests: the JSON constraint applies after the reasoning block
        command += ["--reasoning-parser", reasoning_parser]
    if loras:
        # One flag followed by every NAME=PATH: vLLM keeps only the last of repeated --lora-modules flags.
        command += ["--lora-modules", *(f"{name}={path}" for name, path in sorted(loras.items()))]
    return command


def parse_loras(values: list[str]) -> dict[str, Path]:
    loras = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--lora expects NAME=PATH, got {value!r}")
        loras[name] = Path(path)
    return loras


def precision(model: str, revision: str, adapters: bool) -> str:
    """Weights as served: the checkpoint's quantization config if it has one, otherwise bf16."""
    try:
        from huggingface_hub import hf_hub_download

        config = json.loads(Path(hf_hub_download(model, "config.json", revision=revision)).read_text())
    except Exception:  # noqa: BLE001  (recorded as unknown, never guessed)
        return "unknown"
    quant = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config")
    if quant:
        return f"{quant.get('quant_method', 'quantized')} ({quant.get('format') or quant.get('config_groups', {}) and 'config groups' or 'see config'})"
    return "bf16 base weights (adapters trained with a 4-bit NF4 base)" if adapters else "bf16"


def serving_record(model: str, revision: str, command: list[str], loras: dict[str, Path]) -> dict:
    from transformers import AutoTokenizer

    from modelforge.envinfo import environment, sha256_dir, tokenizer_fingerprint

    adapters = {}
    for name, path in loras.items():
        training = path.parent / "training_record.json"
        adapters[name] = {"path": str(path), "sha256": sha256_dir(path),
                          "training_record_adapter_sha256": (json.loads(training.read_text()).get("adapter_sha256")
                                                             if training.exists() else None)}
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    template = getattr(tokenizer, "chat_template", None) or ""
    return {"model": model, "revision": revision, "command": command, "adapters": adapters,
            "tokenizer": tokenizer_fingerprint(tokenizer), "environment": environment(),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if template else None,
            "serving_precision": precision(model, revision, bool(loras)),
            "environment_variables": SERVE_ENV,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main(model: str, revision: str, loras: dict[str, Path], dry_run: bool = False, **options) -> int:
    command = build_command(model, revision, loras=loras, **options)
    if dry_run:
        print(" ".join([f"{k}={v}" for k, v in SERVE_ENV.items()] + command))
        return 0
    missing = [str(p) for p in loras.values() if not (p / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(f"adapter folders without adapter_config.json: {missing}")
    record = serving_record(model, revision, command, loras)
    out = Path("artifacts/serving")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"serve-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"serving record: {path}\n{' '.join(command)}", flush=True)
    os.environ.update(SERVE_ENV)
    os.execvp(command[0], command)
    return 0  # not reached
