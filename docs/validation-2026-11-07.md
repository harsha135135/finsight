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
  training data rebuilt; gold upper bound unchanged at 0.905.
- Triton needed `Python.h` (Ubuntu's system Python lacks it): uv-managed Python everywhere.
- vLLM: 0.90 GPU memory refused (Windows holds ~1.2 GiB) → 0.87; FlashInfer sampler needs nvcc → PyTorch sampler.
- `conhost --headless` keep-alive launcher started nothing → hidden PowerShell `Start-Process`.
- The WSL clock jumps ~25 s; all durations use monotonic clocks.

Secondary diagnostic, defined and committed before any fine-tuned run: accuracy if quotes may add or drop
currency symbols and thousands separators. Base model: 0.150 (0.101–0.209) vs 0.130 strict, so quote formatting
accounts for 2 points of the base model's failures. Headline scores always use the strict check.

Checkpoint selection for the full run uses validation rows outside `val_shared_250` (360 available; the first
200 after length filtering are used), so the items that compare tuned vs base never influence model selection.

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
