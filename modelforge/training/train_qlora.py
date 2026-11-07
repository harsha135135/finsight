"""QLoRA fine-tuning with Transformers + PEFT + TRL (run on the GPU machine).

The prompt is rendered with the model's own chat template and the same
``enable_thinking`` setting vLLM uses at inference, then trained as a
prompt/completion pair so loss covers only the assistant's JSON answer.

Checkpoints are written every ``--save-steps`` optimizer steps and contain the
adapter, optimizer, scheduler, RNG state and trainer state (global step and log
history); ``--resume`` continues from the newest complete checkpoint, and the
Trainer skips the batches already consumed. Every run records its settings, data
and base-model hashes, environment, adapter-health check, peak VRAM, duration and
GPU energy next to the adapter.

    uv run --extra gpu python -m modelforge.training.train_qlora \
        --model Qwen/Qwen3-4B --revision <hf-commit> \
        --data artifacts/training/finqa-grouped-v1 --out artifacts/adapters/qwen3-4b-finqa-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from modelforge.envinfo import environment, sha256_dir, tokenizer_fingerprint


def _render(tokenizer, example: dict, enable_thinking: bool) -> dict:
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    full = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False,
                                         enable_thinking=enable_thinking)
    if not full.startswith(prompt):
        raise ValueError("chat template renders the prompt differently with and without the answer; "
                         "refusing to train on a mismatched template")
    completion = full[len(prompt):]
    # The template puts a newline after the end-of-turn token. TRL appends the EOS token to any completion
    # that does not end with it, which would teach "<|im_end|>\n<|im_end|>"; end the target at EOS instead.
    eos = tokenizer.eos_token
    if eos and not completion.endswith(eos) and completion.rstrip().endswith(eos):
        completion = completion.rstrip()
    if eos and not completion.endswith(eos):
        raise ValueError(f"rendered completion does not end with the EOS token {eos!r}")
    return {"prompt": prompt, "completion": completion}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--max-steps", type=int, default=-1, help="for smoke and recovery tests")
    parser.add_argument("--save-steps", type=int, default=25, help="checkpoint interval in optimizer steps")
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="evaluation interval (default: same as --save-steps; 0 disables evaluation)")
    parser.add_argument("--eval-limit", type=int, default=None, help="evaluate on the first N validation examples")
    parser.add_argument("--eval-exclude", default="finqa-grouped-comparison:val_shared_250",
                        help="MANIFEST:SELECTION never used for evaluation loss or checkpoint choice, so the "
                             "items that compare tuned vs base stay out of model selection ('' disables)")
    parser.add_argument("--save-total-limit", type=int, default=3, help="0 keeps every checkpoint")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    parser.add_argument("--batch-hash-log", type=Path, default=None, help="recovery test: log micro-batch hashes")
    parser.add_argument("--health-check-after", type=int, default=3)
    return parser.parse_args(argv)


def load_splits(tokenizer, data_dir: Path, max_seq_len: int, enable_thinking: bool) -> dict[str, tuple[list[dict], dict]]:
    """Render every example with the chat template; drop (and count) examples longer than max_seq_len."""
    splits = {}
    for split in ("train", "val"):
        rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        kept, lengths = [], []
        for row in rows:
            rendered = _render(tokenizer, row, enable_thinking)
            n_tokens = len(tokenizer(rendered["prompt"] + rendered["completion"]).input_ids)
            lengths.append(n_tokens)
            if n_tokens <= max_seq_len:
                kept.append({**rendered, "n_tokens": n_tokens, "id": row["id"]})
        ordered = sorted(lengths)
        splits[split] = (kept, {"examples": len(rows), "kept": len(kept), "dropped_too_long": len(rows) - len(kept),
                                "max_tokens": ordered[-1], "p95_tokens": ordered[int(0.95 * (len(ordered) - 1))],
                                "kept_tokens": sum(r["n_tokens"] for r in kept)})
    return splits


def build_trainer(args: argparse.Namespace, tokenizer, train_rows: list[dict], eval_rows: list[dict] | None,
                  callbacks: list | None = None, output_dir: Path | None = None):
    """The one training setup shared by training, the GPU probe and the recovery test."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from modelforge.probe import language_lora_targets, load_text_model

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    model, model_class = load_text_model(args.model, args.revision, quantization)
    targets, coverage = language_lora_targets(model)
    if not coverage["coverage_ok"]:
        raise SystemExit(f"LoRA targets do not cover every decoder layer: {json.dumps(coverage)}")
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    eval_steps = args.save_steps if args.eval_steps is None else args.eval_steps
    evaluate = bool(eval_rows) and eval_steps > 0
    config = SFTConfig(
        output_dir=str(output_dir or (args.out / "checkpoints")), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True,
        max_length=args.max_seq_len, completion_only_loss=True, packing=False,
        save_strategy="steps" if args.save_steps > 0 else "no", save_steps=max(args.save_steps, 1),
        save_total_limit=args.save_total_limit or None,
        eval_strategy="steps" if evaluate else "no", eval_steps=eval_steps if evaluate else None,
        load_best_model_at_end=evaluate and args.save_steps > 0 and eval_steps == args.save_steps,
        metric_for_best_model="eval_loss" if evaluate else None,
        logging_steps=args.logging_steps, report_to=[], seed=args.seed, data_seed=args.seed,
    )
    columns = ("prompt", "completion")
    train_ds = Dataset.from_list([{k: r[k] for k in columns} for r in train_rows])
    # The eval set is prepared whenever rows are given (so the probe can time trainer.evaluate()), even if
    # periodic evaluation is off.
    eval_ds = Dataset.from_list([{k: r[k] for k in columns} for r in eval_rows]) if eval_rows else None
    trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tokenizer, peft_config=lora, callbacks=callbacks or [])
    return trainer, {"model_class": model_class, "lora_targets": targets, "coverage": coverage}


RUN_KEYS = ("model", "revision", "data", "max_seq_len", "epochs", "lr", "lora_r", "lora_alpha", "lora_dropout",
            "batch_size", "grad_accum", "seed", "enable_thinking", "max_steps", "save_steps", "eval_steps",
            "eval_limit", "eval_exclude", "health_check_after")


def run_identity(args: argparse.Namespace, data_manifest: dict) -> dict:
    """Everything that defines the experiment; a resumed session must match it exactly."""
    identity = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k in RUN_KEYS}
    identity.update(train_sha256=data_manifest["train"]["sha256"], val_sha256=data_manifest["val"]["sha256"],
                    prompt_sha256=data_manifest.get("prompt_sha256"))
    return identity


def check_run_identity(out: Path, identity: dict, resume: bool) -> None:
    """A fresh run needs an empty output folder; --resume only continues the same experiment."""
    path = out / "run_identity.json"
    has_checkpoints = latest_checkpoint(out / "checkpoints") is not None
    if resume:
        if path.exists():
            previous = json.loads(path.read_text())
            changed = sorted(k for k in set(previous) | set(identity) if previous.get(k) != identity.get(k))
            if changed:
                raise SystemExit(f"--resume refused: {out} holds a run with different settings: {changed}")
        elif has_checkpoints:
            raise SystemExit(f"--resume refused: {out} has checkpoints but no run_identity.json to verify them")
    elif has_checkpoints or path.exists():
        raise SystemExit(f"{out} already holds a run; use a fresh --out, or --resume to continue that same run")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=1, sort_keys=True) + "\n")


def session_totals(out: Path) -> dict:
    """Sum time and energy over every session, including ones that were killed (last periodic update)."""
    records = [json.loads(f.read_text()) for f in sorted((out / "sessions").glob("*.json"))]
    energy = [r["gpu_energy_joules"] for r in records]
    return {"sessions": len(records),
            "wall_seconds": round(sum(r["wall_seconds"] for r in records), 1),
            "gpu_energy_joules": round(sum(energy), 1) if records and all(e is not None for e in energy) else None,
            "complete": bool(records) and all(r["status"] == "completed" for r in records),
            "note": "an interrupted session counts up to its last update (every 10 optimizer steps and at each save)"}


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Newest checkpoint whose trainer_state.json exists (the Trainer writes it last)."""
    complete = [p for p in checkpoint_dir.glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    return max(complete, key=lambda p: int(p.name.split("-")[1]), default=None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from modelforge.energy import GpuEnergyMeter
    from modelforge.training.callbacks import AdapterHealthCallback, HashingCollator, SessionRecorder

    data_manifest = json.loads((args.data / "data_manifest.json").read_text())
    check_run_identity(args.out, run_identity(args, data_manifest), args.resume)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    splits = load_splits(tokenizer, args.data, args.max_seq_len, args.enable_thinking)
    train_rows, eval_rows = splits["train"][0], splits["val"][0]
    excluded = 0
    if args.eval_exclude:
        from modelforge import splits as manifests
        name, _, selection = args.eval_exclude.partition(":")
        held_out = set(manifests.load(name)["samples"][selection])
        excluded = sum(r["id"] in held_out for r in eval_rows)
        eval_rows = [r for r in eval_rows if r["id"] not in held_out]
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
    eval_ids_sha256 = hashlib.sha256("\n".join(r["id"] for r in eval_rows).encode()).hexdigest()
    health = AdapterHealthCallback(check_after=args.health_check_after, fail_fast=True)
    trainer, info = build_trainer(args, tokenizer, train_rows, eval_rows, callbacks=[health])
    if args.batch_hash_log:
        trainer.data_collator = HashingCollator(trainer.data_collator, args.batch_hash_log)
    resume_from = latest_checkpoint(args.out / "checkpoints") if args.resume else None
    if args.resume and resume_from is None:
        print("no complete checkpoint found; starting from step 0")
    torch.cuda.reset_peak_memory_stats()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.monotonic()  # the WSL wall clock can jump; durations use a monotonic clock
    with GpuEnergyMeter() as meter:
        recorder = SessionRecorder(args.out / "sessions", meter, started_at, started,
                                   str(resume_from) if resume_from else None)
        trainer.add_callback(recorder)
        result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        recorder.write(trainer.state.global_step, "completed")
    reading = meter.reading()
    adapter_dir = args.out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    record = {
        "base_model": args.model, "base_revision": args.revision, **info,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "quantization": "bitsandbytes nf4, double quant, bf16 compute", "data_manifest": data_manifest,
        "tokenizer": tokenizer_fingerprint(tokenizer), "environment": environment(),
        "resumed_from": str(resume_from) if resume_from else None,
        "dataset_lengths": {s: meta for s, (_, meta) in splits.items()}, "eval_examples": len(eval_rows),
        "eval_excluded_examples": excluded, "eval_ids_sha256": eval_ids_sha256,
        "adapter_health": health.report,
        "train_metrics": result.metrics, "log_history": trainer.state.log_history,
        "global_step": trainer.state.global_step, "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_vram_gib": {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                          "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 3)},
        "wall_seconds_this_session": round(time.monotonic() - started, 1),
        "gpu_energy_this_session": {"method": reading.method, "joules": reading.gpu_energy_joules, "notes": reading.notes,
                                    "scope": "GPU only (NVML); excludes CPU, memory, storage, cooling and PSU losses"},
        "adapter_sha256": sha256_dir(adapter_dir),
        "checkpoint_selection": "lowest eval_loss on the held-out validation rows, evaluated every --save-steps "
                                "(frozen before training); generated-answer accuracy is measured afterwards on "
                                "the separate comparison set",
        "all_sessions": session_totals(args.out),
    }
    (args.out / "training_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({k: record[k] for k in ("adapter_sha256", "global_step", "best_checkpoint", "peak_vram_gib",
                                             "all_sessions")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
