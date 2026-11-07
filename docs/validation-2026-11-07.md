# GPU validation runs, 2026-11-07

Short validation runs before any full fine-tuning, on the RTX 4070 12 GB (WSL2, driver 591.86) at
`Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`. No Claude calls (spend $0.00).
Environment: uv-managed CPython 3.12.16, torch 2.14.0+cu130, transformers 5.20.2, peft 0.21.2, trl 1.17.1,
bitsandbytes 0.50.2, vLLM 0.33.0 (`uv.lock` sha256 `7cc379ad…`).

## WSL persistence

- Without a keep-alive, WSL powered off at 05:43:17 UTC, less than a minute after the last Ubuntu window
  closed; tmux, systemd and SSH sessions into WSL do not keep it running.
- With the logon task (`docs/gpu-setup.md` §5): after a Windows sign-out and sign-in with nothing opened, WSL
  came back on a new boot with the task's `/bin/sleep infinity` keep-alive. Over 06:14:08–06:44:24 UTC, with no
  Ubuntu window or Mac session, the boot_id stayed the same, uptime rose from 204 to 2,028 s with no gaps
  (31 one-minute log lines), and the keep-alive count stayed at 1.

## Probe (real trainer, 3 optimizer steps × 16 of the longest examples ≤ 3,072 tokens)

| Measurement | Result |
|---|---|
| 4-bit base weights | 2.56 GiB |
| Training peak VRAM | 3.97 GiB allocated, 4.50 GiB reserved (limit 10.5) |
| Seconds per optimizer step | 36.5, 34.1, 33.9 |
| Training throughput | 1,335 tokens/s |
| Adapter | 252 modules (36 layers × q/k/v/o/gate/up/down), 33.0M trainable parameters; all received gradients and changed |
| Evaluation | 0.59 s per validation example |
| GPU energy (NVML counter works under WSL) | 19.2 kJ for the 3 steps (~183 W) |

Projection for the full run with the answer-v4 data: 1,674 training examples kept (39 longer than 3,072 tokens
dropped), 3.13M tokens, 210 optimizer steps over 2 epochs, about 1.3 h of training (pessimistic: measured on
the longest examples), plus about 2 min per evaluation on a fixed 200-example validation subset.

## Kill-and-resume test (6 steps, save every 2, SIGKILL after checkpoint 4, `--resume`)

All checks passed: checkpoint files present (adapter, optimizer, scheduler, RNG, trainer state); the resumed run
continued at step 5 and ended at step 6; learning rate identical at steps 5–6; the 8 micro-batches after the
resume point byte-identical to the uninterrupted run (no batches re-read); loss within 0.15% and 0.18%; final
adapter weights within 0.74% relative L2.

## Base-model baseline, `finqa-grouped-comparison/val_shared_250` (prompt answer-v4, greedy, guided JSON)

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| Accuracy (answerable) | 0.130 | 0.082–0.187 | 200 |
| Abstention recall (unanswerable) | 0.300 | 0.184–0.431 | 50 |
| False abstention (answerable) | 0.020 | 0.000–0.046 | 200 |
| Invalid output | 0.128 | 0.085–0.172 | 250 |
| Evidence-validation failure | 0.524 | 0.468–0.583 | 250 |
| Operational (9 outputs hit the 1,024-token cap) | 0.036 | 0.015–0.059 | 250 |

Evidence-validation failures (131 items) by subcategory, counted per item (an item can have several): stated
answer disagrees with the executed program 40; figure is in a different passage than the one cited 33; value or
scale disagrees with the quote 28; quote formatting differs from the passage (added `$` or commas) 25; quoted text
not found in the evidence 11; quote is not a single figure 2; mixed units 1. These are kept apart because not all
of them mean the answer contradicts its evidence. Separately, 29 answers pass every check but are wrong.

Latency and throughput (same server):

| Setting | Latency p50 / p95 | First token p50 | Throughput |
|---|---|---|---|
| Mac via SSH tunnel, 1 request at a time (250 items) | 7.4 s / 17.1 s | 228 ms | 52.5 output tok/s |
| Same 50 items: Mac vs PC-local | 7.74 s vs 7.73 s | 215 vs 216 ms | network adds no measurable time |
| PC-local, 8 concurrent (250 items) | 19.6 s / 29.4 s | 10.3 s | 0.40 items/s, 162 output tok/s |

Run agreement: two one-at-a-time runs (Mac, PC) agree on 98% of raw outputs and 100% of outcomes (50 items);
concurrency 8 vs 1 changes 46% of raw outputs and 8% of outcomes (99.6% same correctness), so scored runs use
one request at a time for every model.

## Adapter serving check (vLLM 0.33 LoRA)

- A 6-step adapter from the trainer (`rec6`) and a copy with every LoRA B matrix zeroed (`zero`, same rank-16
  targets) both load in vLLM; adapters must follow a single `--lora-modules` flag.
- `zero` vs the base model on the same server: identical raw output on 20/20 validation questions, so the adapter
  path does not change answers by itself.
- Registering adapters changes the base model's exact text relative to a server without them (35% identical raw
  output, 90% identical outcome, 100% identical correctness on the same 20 questions). The tuned-vs-base comparison
  therefore re-runs the base model on the same server instance that serves the tuned adapter.
- The zero adapter's median latency was 9.2 s vs 8.5 s for the base on that server: adapter compute costs about 8%
  at one request at a time, and is reported as part of the tuned model's latency.

## Probe at a 4,096-token cap

Peak VRAM 4.39 GiB allocated / 5.41 GiB reserved; 1,321 tokens/s; keeps 1,710 of 1,713 training examples
(3 dropped) and every validation example; projected 1.37 h for 2 epochs; all gates passed.

## Issues found and fixed before the baseline

- Operand names had to start with a letter, but the model-facing schema drops patterns and the prompt never said
  so; FinQA's tokenized text (`$ 5.5 billion`) failed support checks for faithful quotes (`$5.5 billion`). Fixed in
  prompt `answer-v4` and the support check (spacing beside symbols ignored; spacing between digits kept);
  training data rebuilt; the gold-converter baseline stayed at 0.905. (This is a baseline, not a ceiling: the tuned
  model later passed 3 of the 19 items the gold converter fails.)
- Triton needed `Python.h` (Ubuntu's system Python lacks it): uv-managed Python everywhere.
- vLLM: 0.90 GPU memory refused (Windows holds ~1.2 GiB) → 0.87; FlashInfer sampler needs nvcc → PyTorch sampler.
- `conhost --headless` keep-alive launcher started nothing → hidden PowerShell `Start-Process`.
- The WSL clock jumps ~25 s; all durations use monotonic clocks.

Secondary diagnostic, defined and committed before any fine-tuned run: accuracy if quotes may add or drop
currency symbols and thousands separators. Base model: 0.150 (0.101–0.209) vs 0.130 strict, so quote formatting
accounts for 2 points of the base model's failures. Headline scores always use the strict check.

Checkpoint selection for the full run used validation rows outside `val_shared_250` (360 available; the first
200 after length filtering). **Correction:** the exclusion matched exact item ids only. The 200 selection rows shared
26 question families (an original and its `::unanswerable` variant on opposite sides) and 87 of their 100 filing
groups with `val_shared_250`, so checkpoint selection was not independent of the comparison set. That the final
checkpoint won does not show the overlap had no effect. v2 splits validation by filing group (see
`evals/records/qwen3-4b-finqa-v1.json` and the v2 manifest).

## Full fine-tune plan (approved 2026-11-07, frozen before training)

- `--max-seq-len 4096 --max-steps 210 --save-steps 30 --eval-limit 200`, other settings at their defaults
  (QLoRA r16/α32/dropout 0.05 on all 7 projections, lr 2e-4 cosine, warmup 3%, batch 1 × accumulation 16, seed 42,
  thinking off), in a fresh output folder without `--resume`; `--resume` is only for recovering this same run, and
  the trainer refuses it if the run's settings or data differ (`run_identity.json`).
- Checkpoint selection: the checkpoint with the lowest validation loss on the 200 held-out validation rows
  (outside `val_shared_250`), evaluated every 30 steps. Generated-answer accuracy is then measured once for the
  selected adapter on the separate 250-question comparison set, with the base model re-run on the same server.
- Time: the token-based estimate is about 1.4 h plus about 15 min of evaluation; the probe's 40–45 s per step on the
  longest examples would mean up to about 2.6 h, so three hours are reserved.
- Each session writes its time and GPU energy every 10 steps and at each save, so a killed session still counts
  and totals are marked incomplete after any interruption.

## Full fine-tune result (training only; accuracy below once measured)

| | Result |
|---|---|
| Run | `artifacts/adapters/qwen3-4b-finqa-v1`, commit `e456670`, one session, no interruptions |
| Steps / epochs | 210 / 1.96 (1,710 examples ≤ 4,096 tokens) |
| Wall time (including 7 evaluations) | 5,322 s (1 h 28.7 min) |
| Held-out validation loss (steps 30…210) | 0.0773, 0.0555, 0.0428, 0.0412, 0.0402, 0.0392, **0.0391** → checkpoint 210 selected |
| Mean training loss | 0.102 (1.75 at step 5) |
| Peak VRAM | 4.40 GiB allocated, 5.41 GiB reserved |
| GPU energy (NVML, GPU only) | 1.023 MJ = 0.284 kWh |
| Adapter | 252 modules, 33.0M parameters, health check ok, sha256 `5f1fafffd63e…` |

## Tuned vs base on `val_shared_250` (same vLLM server, one request at a time, from the Mac)

Paired, filing-clustered 95% CIs (`modelforge compare`; runs `20261107-paired-{tuned,base}-val_shared_250`):

| Metric | Base | Tuned | Tuned − base (95% CI) | n |
|---|---:|---:|---|---:|
| **Accuracy (answerable)** | 0.125 | **0.510** | **+0.385 (+0.318, +0.452)** | 200 |
| Accuracy, lenient quotes (diagnostic) | 0.150 | 0.510 | +0.360 (+0.290, +0.427) | 200 |
| Abstention recall (unanswerable) | 0.260 | 0.320 | +0.060 (−0.060, +0.173) | 50 |
| False abstention (answerable) | 0.020 | 0.005 | −0.015 (−0.043, +0.009) | 200 |
| Invalid output | 0.148 | 0.004 | −0.144 (−0.198, −0.093) | 250 |
| Evidence-validation failure | 0.508 | 0.172 | −0.336 (−0.407, −0.264) | 250 |
| Operational failure | 0.020 | 0.004 | −0.016 (−0.032, −0.004) | 250 |

Answerable items: 79 fixed by tuning, 2 broken, 23 correct in both, 96 wrong in both. The gain is not quote
formatting (lenient accuracy moves almost as much as strict). Remaining evidence-validation failures in the tuned
model are mostly "stated answer disagrees with the executed program" (31 items); only 1 of the 26 answerable
mismatches states the gold answer. **Correction:** these mismatches show an inconsistency between the stated answer
and the program; they do not rule out memorization, and arithmetic is not shown to be the only cause.

Weakness: the tuned model answered 24 of the 50 unanswerable variants with outputs that pass validation. Two of
those reproduce the original gold answer (the removed figure is derivable from the remaining evidence, a label flaw
in the variant construction); the other 22 are confident wrong answers built from figures that are present. The
validator cannot catch these, so abstention needs more or harder abstention training data.

Latency and throughput (same server): median 3.8 s vs 7.3 s, p95 4.5 s vs 12.3 s; output tokens 43.8k vs 101.9k
for the 250 items (compact JSON with only the needed operands); decoding 46.5 vs 52.4 tokens/s (adapter compute
about 11% per token); 0.27 vs 0.13 items/s at one request at a time.

## Corrections (2026-11-07, after review)

- 0.905 is the **gold-converter baseline** on these 200 answerable items, not an achievable-score ceiling; an earlier
  "56% of what's achievable" statement in a status report is withdrawn. The result is 51.0% accuracy.
- Checkpoint selection overlapped the comparison set (26 question families, 87 of 100 filing groups); see above.
- Answer/program mismatches establish inconsistency only; memorization is not ruled out.
- `val_shared_250` has guided decisions and is development data. v1 is frozen in
  `evals/records/qwen3-4b-finqa-v1.json` (artifact hashes; `modelforge record verify` regenerates every scorecard).

## Rescored with audited labels (v1 unchanged; separately versioned)

All 50 of v1's `val_shared_250` negatives were reviewed by hand (`evals/labels/finqa-unanswerable-audit-valshared250-v1.json`;
model outputs never consulted): 28 verified unanswerable, 19 derivable (mostly accounting identities: totals minus
components, roll-forwards, shares summing to 100%), 3 ambiguous. Excluding the 22 invalid negatives
(`modelforge relabel`; record `evals/records/qwen3-4b-finqa-v1-relabeled-v1.json`):

| Metric (228 items) | Base | Tuned | Tuned − base (95% CI) |
|---|---:|---:|---|
| Accuracy (200 answerable, unchanged) | 0.125 | 0.510 | +0.385 (+0.318, +0.452) |
| Abstention recall (28 verified negatives) | 0.393 | 0.429 | +0.036 (−0.115, +0.179) |
| Evidence-validation failure | 0.500 | 0.175 | −0.325 (−0.392, −0.251) |

Diagnostics (`modelforge diagnose`): **program accuracy**, the executed program's value correct whatever the stated
answer says, is 0.135 base vs **0.550 tuned** (paired +0.415); FinSight takes final numbers from the calculator, so this
isolates operand and formula choice. Arithmetic consistency (stated answer agrees with the executed program) is 0.648
base (n=91) vs 0.863 tuned (n=190). Abstention still does not improve on verified negatives.

## v1 checkpoint re-selection on the corrected selection set

Retained checkpoints re-evaluated on `val_select` rows from `finqa-grouped-v2` (237 rows from 65 filing groups disjoint
from `val_dev`, `val_shared_250`'s groups and test; 22 hand-verified negatives; `modelforge.training.reselect`):
checkpoint-150 0.045927, checkpoint-180 0.045493, **checkpoint-210 0.045468**: the same checkpoint is selected. This
is partial evidence: checkpoints 30–120 were deleted by `save_total_limit` and cannot be compared.
