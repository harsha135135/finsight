"""Build the final freeze file for the sealed evaluations (run only after the configuration is approved).

    uv run python scripts/make_freeze.py --out evals/freeze-final-v2.json

The freeze records what the sealed runs must use: models and revisions, serving and generation settings, the answer
contracts and prompts (with hashes), validator and scorer versions, retrieval version, the frozen rules, and the sha256
of every file that implements them. It is committed once and never edited; held-out inputs go into a second file that
cites this file's sha256 (docs/evaluation-rules.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CODE = ["finsight/qa.py", "finsight/contracts/qa.py", "finsight/reasoning/qa_prompt.py", "finsight/reasoning/qwen.py",
        "finsight/normalize.py", "finsight/calc.py", "finsight/findings.py", "finsight/retrieval/search.py",
        "finsight/ingestion/pdf.py", "modelforge/qa_score.py", "modelforge/qa_eval.py", "modelforge/fb_retrieval.py",
        "modelforge/fiscal.py", "modelforge/grading.py", "modelforge/claude_qa_batch.py", "modelforge/claude_batch.py",
        "modelforge/runner.py", "modelforge/prompts.py", "modelforge/audit.py", "modelforge/splits.py",
        "modelforge/datasets/financebench.py", "modelforge/serve.py", "modelforge/stats.py"]
DATA = ["evals/manifests/financebench-v1.json", "evals/manifests/finqa-grouped-comparison.json",
        "evals/manifests/finqa-grouped-v2.json", "docs/evaluation-rules.md",
        "evals/financebench/fiscal-years-dev-v1.json", "evals/financebench/fiscal-years-dev-derived-v1.json",
        "evals/financebench/evidence-audit-dev-v1.json", "evals/financebench/retrieval-dev-pdf2-text-3.json"]


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build() -> dict:
    from finsight.reasoning.qa_prompt import QA_PROMPT_VERSION, SYSTEM, claude_schema, qa_schema
    from modelforge import qa_score
    from modelforge.prompts import PROMPT_VERSION as FINQA_PROMPT

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if dirty:
        raise SystemExit("commit everything first: the freeze must describe a committed tree")
    h = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
    return {
        "name": "freeze-final-v2", "date": date.today().isoformat(), "commit": commit,
        "financebench": {
            "split": "held_out (75 questions, 16 companies; sealed until authorized)",
            "modes": ["supplied evidence", "full document"],
            "contract": "finsight-qa-v1", "prompt_version": QA_PROMPT_VERSION, "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "schema_sha256": {"local": h(qa_schema()), "claude": h(claude_schema())},
            "scorer": qa_score.CURRENT, "headline_validation": "strict (lenient also reported)",
            "retrieval": {"parser": "pdf-2", "version": "text-3", "limit": 25, "char_budget": 8000},
            "local_model": {"model": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "precision": "bf16",
                            "serving": "vLLM 0.33.0; modelforge serve --compact-json --max-model-len 6656 (no adapters needed)",
                            "generation": {"max_tokens": 1536, "thinking": False, "decoding": "greedy, guided JSON"}},
            "reference_model": {"model": "claude-haiku-5-5", "api": "Message Batches", "effort": "medium", "max_tokens": 5600,
                                "cap_usd_per_run": "0.25 (worst case must fit; exact count before submission; approval required)"},
        },
        "finqa": {
            "selection": "finqa-grouped-comparison / test_shared_300 (sealed); unanswerable variants kept only if the test-pool audit verifies them",
            "protocol": FINQA_PROMPT, "local_models": {"base": "Qwen/Qwen3-4B@1cfa9a72", "v1": "qwen3-4b-finqa-v1 adapter (highest val_dev accuracy)"},
            "serving": "modelforge serve defaults (max-model-len 5120), the FinQA protocol the development results used",
            "reference_model": {"model": "claude-haiku-5-5", "api": "Message Batches", "effort": "medium", "max_tokens": 2600,
                                "cap_usd_per_run": "0.25 (approval required)"},
        },
        "rules": "docs/evaluation-rules.md (opening order, fiscal years, exclusions, headline measures, narrative grading)",
        "code_sha256": {p: sha(p) for p in CODE}, "data_sha256": {p: sha(p) for p in DATA},
        "sealed": "neither the FinanceBench held-out half nor the FinQA test pool has been read",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="write here (omit to print)")
    args = parser.parse_args()
    data = build()
    body = json.dumps(data, indent=1, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"{args.out} exists; a freeze is never rewritten")
        args.out.write_text(body)
    print(body)
