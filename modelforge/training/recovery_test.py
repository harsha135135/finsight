"""Kill-and-resume test for QLoRA training (run on the GPU machine).

Run A trains uninterrupted for 6 optimizer steps, saving every 2. Run B uses the
same settings, is killed with SIGKILL once checkpoint-4 is complete, then resumed
with ``--resume`` to step 6. Evaluation is off so the batch-hash log sees only
training batches.

Exact checks: checkpoint-4 holds the adapter, optimizer, scheduler, RNG and trainer
state; the resumed run continues at step 5 and ends at step 6; the learning rate at
steps 5-6 equals run A's; the micro-batches after the resume point are byte-identical
to run A's (data position). Tolerance checks (GPU kernels are not bit-deterministic):
loss at steps 5-6 and the final adapter weights.

    uv run --extra gpu python -m modelforge.training.recovery_test --model Qwen/Qwen3-4B --revision <sha>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                  "rng_state.pth", "trainer_state.json")
TOTAL_STEPS, SAVE_EVERY, KILL_AFTER = 6, 2, 4


def _losses(record: dict) -> dict[int, dict]:
    return {h["step"]: h for h in record["log_history"] if "loss" in h and "step" in h}


def _hashes(path: Path) -> list[str]:
    return [json.loads(line)["input_ids_sha"] for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def compare(record_a: dict, record_b: dict, hashes_a: list[str], hashes_resumed: list[str], grad_accum: int,
            checkpoint_files: list[str], loss_rtol: float = 2e-2) -> dict:
    """Pure comparison of the two runs (unit-tested without a GPU)."""
    logs_a, logs_b = _losses(record_a), _losses(record_b)
    after = list(range(KILL_AFTER + 1, TOTAL_STEPS + 1))
    lr_equal = all(s in logs_a and s in logs_b and logs_a[s]["learning_rate"] == logs_b[s]["learning_rate"] for s in after)
    loss_rel = {s: abs(logs_a[s]["loss"] - logs_b[s]["loss"]) / max(abs(logs_a[s]["loss"]), 1e-12)
                for s in after if s in logs_a and s in logs_b}
    expected = hashes_a[KILL_AFTER * grad_accum: TOTAL_STEPS * grad_accum]
    offset = next((i for i in range(len(hashes_resumed)) if hashes_resumed[i:i + len(expected)] == expected), None)
    checks = {
        "checkpoint_files_present": all(f in checkpoint_files for f in REQUIRED_FILES),
        "resumed_from_kill_checkpoint": str(record_b.get("resumed_from") or "").endswith(f"checkpoint-{KILL_AFTER}"),
        "global_step_continues": record_b.get("global_step") == TOTAL_STEPS and all(s in logs_b for s in after),
        "learning_rate_identical": lr_equal,
        "data_position_identical": bool(expected) and offset is not None,
        "loss_within_tolerance": len(loss_rel) == len(after) and all(v <= loss_rtol for v in loss_rel.values()),
    }
    return {"checks": checks, "passed": all(checks.values()), "loss_relative_difference": loss_rel,
            "batches_collated_before_resume_point": offset, "expected_batches_compared": len(expected),
            "missing_checkpoint_files": [f for f in REQUIRED_FILES if f not in checkpoint_files]}


def _adapter_difference(a: Path, b: Path) -> dict:
    from safetensors.torch import load_file
    wa, wb = load_file(str(a / "adapter_model.safetensors")), load_file(str(b / "adapter_model.safetensors"))
    if wa.keys() != wb.keys():
        return {"same_tensors": False}
    max_abs = max(float((wa[k].float() - wb[k].float()).abs().max()) for k in wa)
    num = sum(float((wa[k].float() - wb[k].float()).pow(2).sum()) for k in wa) ** 0.5
    den = sum(float(wa[k].float().pow(2).sum()) for k in wa) ** 0.5
    return {"same_tensors": True, "max_abs_difference": max_abs, "relative_l2_difference": num / max(den, 1e-12)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", default="artifacts/training/finqa-grouped-v1")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument("--out", type=Path, default=Path(f"artifacts/recovery/{time.strftime('%Y%m%dT%H%M%S')}"))
    parser.add_argument("--weights-rtol", type=float, default=2e-2)
    args = parser.parse_args(argv)

    def command(out: Path, hash_log: Path, *extra: str) -> list[str]:
        return [sys.executable, "-m", "modelforge.training.train_qlora", "--model", args.model,
                "--revision", args.revision, "--data", args.data, "--out", str(out),
                "--max-steps", str(TOTAL_STEPS), "--save-steps", str(SAVE_EVERY), "--eval-steps", "0",
                "--save-total-limit", "0", "--logging-steps", "1", "--grad-accum", str(args.grad_accum),
                "--max-seq-len", str(args.max_seq_len), "--batch-hash-log", str(hash_log), *extra]

    run_a, run_b = args.out / "uninterrupted", args.out / "interrupted"
    for d in (run_a, run_b):
        d.mkdir(parents=True, exist_ok=True)
    print("run A: uninterrupted", flush=True)
    subprocess.run(command(run_a, run_a / "batches.jsonl"), check=True)

    print(f"run B: will be killed after checkpoint-{KILL_AFTER} is complete", flush=True)
    marker = run_b / "checkpoints" / f"checkpoint-{KILL_AFTER}" / "trainer_state.json"
    proc = subprocess.Popen(command(run_b, run_b / "batches_before_kill.jsonl"), start_new_session=True)
    deadline = time.monotonic() + 3600
    while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    killed = proc.poll() is None
    if killed:
        time.sleep(0.5)
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    if not killed or not marker.exists():
        print("the run finished or failed before it could be killed; the test is invalid", flush=True)
        return 1
    if (run_b / "checkpoints" / f"checkpoint-{TOTAL_STEPS}").exists():
        print("a later checkpoint was written before the kill; the test is invalid", flush=True)
        return 1
    print("run B: resuming", flush=True)
    subprocess.run(command(run_b, run_b / "batches_resumed.jsonl", "--resume"), check=True)

    record_a = json.loads((run_a / "training_record.json").read_text())
    record_b = json.loads((run_b / "training_record.json").read_text())
    checkpoint_files = sorted(p.name for p in (run_b / "checkpoints" / f"checkpoint-{KILL_AFTER}").iterdir())
    report = compare(record_a, record_b, _hashes(run_a / "batches.jsonl"), _hashes(run_b / "batches_resumed.jsonl"),
                     args.grad_accum, checkpoint_files)
    report["final_adapter"] = _adapter_difference(run_a / "adapter", run_b / "adapter")
    report["checks"]["final_adapter_within_tolerance"] = bool(
        report["final_adapter"].get("same_tensors")
        and report["final_adapter"]["relative_l2_difference"] <= args.weights_rtol)
    report["passed"] = all(report["checks"].values())
    report["settings"] = {"total_steps": TOTAL_STEPS, "save_every": SAVE_EVERY, "killed_after_checkpoint": KILL_AFTER,
                          "grad_accum": args.grad_accum, "max_seq_len": args.max_seq_len, "signal": "SIGKILL"}
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
