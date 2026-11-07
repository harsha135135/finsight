"""Write a copy of a trained LoRA adapter with every B matrix set to zero (run on the GPU machine).

With B = 0 the adapter adds exactly nothing, so serving it must reproduce the base
model's answers. Used to check that the serving path for adapters (vLLM LoRA) does not
by itself change outputs before any tuned-vs-base comparison. The copy keeps the
source adapter's target modules, rank and config.

    uv run --extra gpu python -m modelforge.training.zero_adapter SOURCE_ADAPTER_DIR OUT_DIR
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    source, out = (Path(a) for a in (argv or sys.argv[1:]))
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(source / "adapter_model.safetensors"))
    zeroed = 0
    for name in tensors:
        if "lora_B" in name:
            tensors[name] = tensors[name].clone().zero_()
            zeroed += 1
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    shutil.copy(source / "adapter_config.json", out / "adapter_config.json")
    print(f"{zeroed} lora_B tensors zeroed out of {len(tensors)}; written to {out}")
    return 0 if zeroed else 1


if __name__ == "__main__":
    raise SystemExit(main())
