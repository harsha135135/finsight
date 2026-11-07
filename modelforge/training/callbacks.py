"""Trainer callbacks and collator wrappers used by training, the probe and the recovery test (GPU machine only)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


def lora_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad}


class AdapterHealthCallback(TrainerCallback):
    """Checks that the intended adapter modules are attached, receive gradients and update.

    LoRA's B matrices start at zero, so A's gradient is exactly zero on the first
    optimizer step. The check therefore requires a gradient tensor (not None) on the
    first step, and a non-zero gradient plus a changed value for every LoRA tensor
    by ``check_after`` optimizer steps (default 3).
    """

    def __init__(self, check_after: int = 3, fail_fast: bool = True) -> None:
        self.check_after = check_after
        self.fail_fast = fail_fast
        self.initial: dict[str, torch.Tensor] = {}
        self.nonzero_grad: dict[str, bool] = {}
        self.missing_grad_first_step: list[str] = []
        self.optimizer_steps = 0
        self.report: dict = {"status": "not_checked"}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        params = lora_parameters(model)
        self.initial = {n: p.detach().float().cpu().clone() for n, p in params.items()}
        self.nonzero_grad = {n: False for n in params}
        modules = sorted({n.split(".lora_")[0] for n in params})
        self.report = {"status": "pending", "lora_modules": len(modules), "lora_module_names": modules,
                       "lora_tensors": len(params), "trainable_parameters": sum(p.numel() for p in params.values()),
                       "all_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if not params:
            self._finish(["no trainable LoRA parameters are attached"])

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.report.get("status") != "pending":
            return
        for name, p in lora_parameters(model).items():
            if p.grad is None:
                if self.optimizer_steps == 0:
                    self.missing_grad_first_step.append(name)
            elif not self.nonzero_grad.get(name) and bool(p.grad.detach().abs().max() > 0):
                self.nonzero_grad[name] = True

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.report.get("status") != "pending":
            return
        self.optimizer_steps += 1
        if self.optimizer_steps < self.check_after:
            return
        params = lora_parameters(model)
        unchanged = [n for n, p in params.items() if torch.equal(p.detach().float().cpu(), self.initial[n])]
        no_grad = [n for n, seen in self.nonzero_grad.items() if not seen]
        problems = []
        if self.missing_grad_first_step:
            problems.append(f"{len(self.missing_grad_first_step)} LoRA tensors had no gradient at the first step")
        if no_grad:
            problems.append(f"{len(no_grad)} LoRA tensors never received a non-zero gradient")
        if unchanged:
            problems.append(f"{len(unchanged)} LoRA tensors did not change")
        self.report.update(missing_grad_first_step=self.missing_grad_first_step[:20], never_nonzero_grad=no_grad[:20],
                           unchanged=unchanged[:20], checked_after_optimizer_steps=self.optimizer_steps)
        self._finish(problems)

    def on_train_end(self, args, state, control, **kwargs):
        if self.report.get("status") == "pending":
            # e.g. a resumed session that ran fewer optimizer steps than the check needs
            self.report.update(status="incomplete", checked_after_optimizer_steps=self.optimizer_steps,
                               problems=[f"only {self.optimizer_steps} optimizer step(s) this session; "
                                         f"the check needs {self.check_after}"])
            self.initial = {}

    def _finish(self, problems: list[str]) -> None:
        self.report["status"] = "ok" if not problems else "failed"
        self.report["problems"] = problems
        self.initial = {}
        if problems and self.fail_fast:
            raise RuntimeError("adapter health check failed: " + "; ".join(problems))


class StepTimer(TrainerCallback):
    """Wall-clock seconds per optimizer step (includes gradient accumulation)."""

    def __init__(self) -> None:
        self.step_seconds: list[float] = []
        self._t = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self._t is None:
            self._t = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.step_seconds.append(now - self._t)
        self._t = now


class HashingCollator:
    """Wraps a data collator and appends a hash of each micro-batch's input_ids to a file.

    Used by the recovery test to prove that a resumed run sees exactly the batches an
    uninterrupted run sees after the resume point. Evaluation must be disabled while it
    is active, because the Trainer uses the same collator for evaluation batches.
    """

    def __init__(self, inner, path: Path) -> None:
        self.inner = inner
        self.path = path

    def __call__(self, features):
        batch = self.inner(features)
        digest = hashlib.sha256(batch["input_ids"].detach().cpu().numpy().tobytes()).hexdigest()[:16]
        with self.path.open("a") as handle:
            handle.write(json.dumps({"input_ids_sha": digest, "shape": list(batch["input_ids"].shape)}) + "\n")
        return batch
