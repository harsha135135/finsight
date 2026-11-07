"""Re-run checkpoint selection for a finished run on a corrected selection set (GPU machine).

Builds the trainer once with ``build_trainer`` (identical tokenization and loss masking to
training), then loads each retained checkpoint's adapter weights and measures evaluation loss.
Only retained checkpoints can be compared, so the result is partial evidence when earlier
checkpoints were deleted.

    uv run --extra gpu python -m modelforge.training.reselect --model Qwen/Qwen3-4B --revision <sha> \
        --data artifacts/training/finqa-grouped-v2-a20 \
        --checkpoints artifacts/adapters/qwen3-4b-finqa-v1/checkpoints --out artifacts/adapters/qwen3-4b-finqa-v1/reselection-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from modelforge.training.train_qlora import build_trainer, latest_checkpoint, load_splits

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    val_rows, val_info = load_splits(tokenizer, args.data, args.max_seq_len, enable_thinking=False)["val"]
    settings = Namespace(model=args.model, revision=args.revision, out=args.out.parent / "reselect-tmp",
                         max_seq_len=args.max_seq_len, epochs=1.0, lr=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                         batch_size=1, grad_accum=1, seed=42, max_steps=1, save_steps=0, eval_steps=0,
                         save_total_limit=0, logging_steps=1)
    trainer, _ = build_trainer(settings, tokenizer, val_rows[:2], val_rows, output_dir=settings.out)
    checkpoints = sorted((p for p in args.checkpoints.glob("checkpoint-*") if (p / "adapter_model.safetensors").exists()),
                         key=lambda p: int(p.name.split("-")[1]))
    losses = {}
    for checkpoint in checkpoints:
        set_peft_model_state_dict(trainer.model, load_file(str(checkpoint / "adapter_model.safetensors")))
        losses[checkpoint.name] = round(trainer.evaluate()["eval_loss"], 6)
        print(checkpoint.name, losses[checkpoint.name], flush=True)
    data_manifest = json.loads((args.data / "data_manifest.json").read_text())
    report = {"checkpoints_compared": list(losses), "eval_loss": losses, "selected": min(losses, key=losses.get),
              "selection_rows": val_info, "selection_rows_sha256": data_manifest["val"]["sha256"],
              "selection_source": data_manifest["val"].get("source"), "data_manifest_sha256":
              hashlib.sha256((args.data / "data_manifest.json").read_bytes()).hexdigest(),
              "originally_selected": latest_checkpoint(args.checkpoints).name if latest_checkpoint(args.checkpoints) else None,
              "limitation": "only retained checkpoints are compared; earlier checkpoints were deleted by save_total_limit"}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
