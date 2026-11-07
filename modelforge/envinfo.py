"""Hardware and software facts recorded with every probe, training run, serving session and evaluation."""

from __future__ import annotations

import hashlib
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "vllm", "httpx", "pydantic")


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def nvidia_smi_path() -> str:
    """WSL keeps nvidia-smi in /usr/lib/wsl/lib, which SSH sessions do not put on PATH."""
    return shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists()
                                          else "nvidia-smi")


def _nvidia_smi() -> str | None:
    try:
        return subprocess.run([nvidia_smi_path(), "--query-gpu=name,driver_version,memory.total,power.limit",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except Exception:  # noqa: BLE001 - absence is a fact to record, not an error
        return None


def environment(root: Path = Path(".")) -> dict:
    proc_version = Path("/proc/version")
    record = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kernel": platform.release(),
        "wsl": proc_version.exists() and "microsoft" in proc_version.read_text().lower(),
        "host": platform.node(),
        "gpu": _nvidia_smi(),
        "packages": {name: _version(name) for name in PACKAGES},
        "uv_lock_sha256": sha256_file(root / "uv.lock"),
        "code_commit": (root / "BUILD_COMMIT").read_text().strip() if (root / "BUILD_COMMIT").exists() else None,
    }
    try:
        import torch
        record["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            record["cuda_device"] = torch.cuda.get_device_name(0)
            record["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except ImportError:
        pass
    return record


def tokenizer_fingerprint(tokenizer) -> dict:
    """Hashes of the chat template and vocabulary, so baseline and tuned runs can prove they match."""
    template = tokenizer.chat_template or ""
    vocab = sorted(tokenizer.get_vocab().items())
    return {"chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "vocab_sha256": hashlib.sha256(repr(vocab).encode()).hexdigest(),
            "eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token}
