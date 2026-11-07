# FinSight

**Autonomous financial research with traceable answers, plus ModelForge for reproducible fine-tuning and evaluation.**

FinSight answers financial questions and writes research reports from evidence: SEC filings, XBRL facts and local documents. Every figure is quoted from a cited passage, every calculation runs in a Decimal interpreter rather than in the model's head, and every result carries a reason code: answered, abstained for missing evidence, invalid model output, contradiction with the evidence, or operational failure.

ModelForge measures how much domain fine-tuning improves a local Qwen model, and when its quality, cost and latency justify using it instead of Claude.

> **Status: under active rebuild** (branch `rebuild`). Results tables stay empty until measured.

## What works today

| Area | State |
|---|---|
| Shared contracts (`finsight/contracts`) | Strict Pydantic models: unknown fields are rejected and decimals serialise as strings. |
| Calculator (`finsight/calc.py`) | Safe interpreter, no `eval()`. Reproduces **8,281 / 8,281** FinQA gold programs (`modelforge calc-suite`). This measures the calculator, not a model. |
| Validation (`finsight/validation.py`) | Citation, operand-support, unit, period and answer-consistency checks with distinct reason codes. |
| FinQA splits (`modelforge/splits.py`) | Frozen, hashed manifests. The official split shares 243 of 278 test filings with train, so `finqa-grouped` keeps the official test set and removes shared filings and passages from train and dev. |
| Evaluation runner | Resumable runs, filing-clustered bootstrap CIs, deterministic scorecards, demo and vLLM-Qwen adapters. |
| QLoRA training (`modelforge/training`) | Data built from the frozen train split only. Base model `Qwen/Qwen3-4B` (pinned revision). Checkpoints every 25 steps with resume, an adapter-health check, and a kill-and-resume test. |
| GPU workflow | Mac → WSL2 over SSH (verified commands, rsync and tunnels); probe, recovery test, vLLM serving on 127.0.0.1. See [`docs/gpu-setup.md`](docs/gpu-setup.md). |
| Comparison sets | `finqa-grouped-comparison`: 300 shared test items (50 declared unanswerable) for all models, plus the expanded 1,247-item test set for local models only. Frozen before any test inference. |

| Research application | FastAPI (`/research-jobs`, status, report, evidence, cancel) over Postgres; a LangGraph worker with checkpoints on its dedicated session, ownership tokens and a supervisor for frozen workers; SEC EDGAR ingestion (filing by fiscal-year end, as-reported XBRL facts, stored originals); table-aware parsing and metric+period retrieval; the `finsight-report-v1` contract with per-finding validation; Qwen, Claude and recorded providers with one Postgres spend ledger. |

Still to come: the real-filing demo run (needs `FINSIGHT_SEC_USER_AGENT`), FinanceBench in ModelForge, Docker packaging and CI. See `docs/` and the plan.

## Quick start (offline, no API key)

```sh
uv sync --extra app
uv run pytest -q
uv run modelforge calc-suite                      # downloads FinQA at a pinned commit
uv run modelforge splits build && uv run modelforge splits verify
uv run modelforge run --adapter gold --selection val_200    # gold-converter baseline
```

GPU machine setup (Windows + WSL2 + SSH): [`docs/gpu-setup.md`](docs/gpu-setup.md).

## Data and licences

FinQA (MIT) is downloaded at a pinned commit into `~/.cache/finsight/datasets`; it is never committed. FinanceBench (CC-BY-NC-4.0) is evaluation-only and is never used for training. Claude outputs are never used as training targets.
