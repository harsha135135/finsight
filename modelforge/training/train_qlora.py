"""QLoRA fine-tuning with Transformers + PEFT + TRL (run on the GPU machine).

The prompt is rendered with the model's own chat template and the same
``enable_thinking`` setting vLLM uses at inference, then trained as a
prompt/completion pair so loss covers only the assistant's JSON answer.
Every run records its settings, data and base-model hashes, peak VRAM,
duration and GPU energy next to the adapter.

    uv run --extra gpu python -m modelforge.training.train_qlora \
        --model Qwen/Qwen3.5-4B --revision <hf-commit> \
        --data artifacts/training/finqa-grouped-v1 --out artifacts/adapters/qwen35-4b-finqa-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path


def _sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _render(tokenizer, example: dict, enable_thinking: bool) -> dict:
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    full = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False,
                                         enable_thinking=enable_thinking)
    if not full.startswith(prompt):
        raise ValueError("chat template renders the prompt differently with and without the answer; "
                         "refusing to train on a mismatched template")
    return {"prompt": prompt, "completion": full[len(prompt):]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Hugging Face commit hash of the base checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", help="must match the serving setting")
    parser.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from modelforge.energy import GpuEnergyMeter
    from modelforge.probe import load_text_model

    args.out.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads((args.data / "data_manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    splits = {}
    for split in ("train", "val"):
        rows = [json.loads(line) for line in (args.data / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        rendered = [_render(tokenizer, row, args.enable_thinking) for row in rows]
        lengths = [len(tokenizer(r["prompt"] + r["completion"]).input_ids) for r in rendered]
        kept = [r for r, n in zip(rendered, lengths) if n <= args.max_seq_len]
        splits[split] = (Dataset.from_list(kept), {"examples": len(rows), "kept": len(kept),
                                                    "dropped_too_long": len(rows) - len(kept),
                                                    "max_tokens": max(lengths), "p95_tokens": sorted(lengths)[int(0.95 * (len(lengths) - 1))]})

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    model, model_class = load_text_model(args.model, args.revision, quantization)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    config = SFTConfig(
        output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True, bf16=True,
        max_length=args.max_seq_len, completion_only_loss=True, packing=False,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", logging_steps=10, report_to=[], seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=splits["train"][0], eval_dataset=splits["val"][0],
                         processing_class=tokenizer, peft_config=lora)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with GpuEnergyMeter() as meter:
        result = trainer.train()
    reading = meter.reading()
    adapter_dir = args.out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    record = {
        "base_model": args.model, "base_revision": args.revision, "model_class": model_class,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "quantization": "bitsandbytes nf4, double quant, bf16 compute", "data_manifest": data_manifest,
        "dataset_lengths": {s: info for s, (_, info) in splits.items()},
        "train_metrics": result.metrics, "eval_history": [h for h in trainer.state.log_history if "eval_loss" in h],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "wall_seconds": round(time.time() - started, 1),
        "gpu_energy": {"method": reading.method, "joules": reading.gpu_energy_joules, "notes": reading.notes,
                       "scope": "GPU only (NVML); excludes CPU, memory, storage, cooling and PSU losses"},
        "adapter_sha256": _sha256_dir(adapter_dir), "host": platform.node(),
        "versions": {m: __import__(m).__version__ for m in ("torch", "transformers", "peft", "trl", "bitsandbytes")},
    }
    (args.out / "training_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({k: record[k] for k in ("adapter_sha256", "peak_vram_gib", "wall_seconds", "gpu_energy")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
