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
                  loras: dict[str, Path] | None = None) -> list[str]:
    executable = shutil.which("vllm")
    base = [executable] if executable else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    command = base + [
        "serve", model, "--revision", revision, "--tokenizer-revision", revision,
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--seed", "0", "--generation-config", "vllm",
        "--enable-lora", "--max-lora-rank", str(max_lora_rank), "--max-loras", "1",
    ]
    for name, path in sorted((loras or {}).items()):
        command += ["--lora-modules", f"{name}={path}"]
    return command


def parse_loras(values: list[str]) -> dict[str, Path]:
    loras = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--lora expects NAME=PATH, got {value!r}")
        loras[name] = Path(path)
    return loras


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
    return {"model": model, "revision": revision, "command": command, "adapters": adapters,
            "tokenizer": tokenizer_fingerprint(tokenizer), "environment": environment(),
            "serving_precision": "bf16 base weights (adapters trained with a 4-bit NF4 base)",
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
