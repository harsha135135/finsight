"""ModelForge command line: splits, the calculator suite, evaluation runs, reports and the GPU probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = Path("artifacts/runs")


def _calc_suite(revision: str) -> dict:
    from finsight.calc import CalcError, execute
    from modelforge.datasets import finqa

    result = {}
    for split in finqa.SPLITS:
        counts: Counter = Counter()
        for example in finqa.load_raw(split, revision):
            qa = example["qa"]
            try:
                steps, rows = finqa.parse_program(qa["program"], example["table"])
                value = execute(steps, {}, rows)[-1]
            except (finqa.FinQAProgramError, CalcError):
                counts["error"] += 1
                continue
            counts["match" if finqa.matches_gold(value, finqa.gold_text(qa["exe_ans"])) else "mismatch"] += 1
        total = sum(counts.values())
        result[split] = {**counts, "total": total, "accuracy": round(counts["match"] / total, 6)}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelforge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("splits", help="build or verify frozen manifests")
    sp.add_argument("action", choices=["build", "verify", "build-comparison", "build-v2"])
    sp.add_argument("--name", default="finqa-grouped")
    cs = sub.add_parser("calc-suite", help="execute FinQA gold programs (measures the calculator, not a model)")
    cs.add_argument("--revision", default=None)
    rp = sub.add_parser("run", help="evaluate an adapter on frozen manifest items")
    rp.add_argument("--adapter", required=True, choices=["gold", "abstain", "qwen"])
    rp.add_argument("--manifest", default="finqa-grouped")
    rp.add_argument("--selection", default="val_200", help="a manifest sample (e.g. val_200) or split (val, test)")
    rp.add_argument("--limit", type=int)
    rp.add_argument("--model", help="vLLM model or LoRA adapter name (qwen)")
    rp.add_argument("--label", help="config id recorded in the scorecard")
    rp.add_argument("--base-url", default="http://127.0.0.1:8000")
    rp.add_argument("--max-tokens", type=int, default=1024, help="qwen: output cap (prompt + cap must fit max-model-len)")
    rp.add_argument("--concurrency", type=int, default=1, help="parallel requests (1 for latency runs)")
    rp.add_argument("--out", type=Path)
    rep = sub.add_parser("report", help="regenerate a scorecard from stored predictions")
    rep.add_argument("run_dir", type=Path)
    ag = sub.add_parser("agree", help="agreement between two runs over their common items")
    ag.add_argument("run_a", type=Path)
    ag.add_argument("run_b", type=Path)
    cp = sub.add_parser("compare", help="paired comparison of two runs on the same items (A minus B)")
    cp.add_argument("run_a", type=Path)
    cp.add_argument("run_b", type=Path)
    dg = sub.add_parser("diagnose", help="program accuracy and arithmetic consistency (diagnostics.json)")
    dg.add_argument("run_dir", type=Path)
    dg.add_argument("--baseline", type=Path)
    rl = sub.add_parser("relabel", help="copy a run with audited labels (unverified negatives excluded)")
    rl.add_argument("run_dir", type=Path)
    rl.add_argument("--labels", type=Path, required=True)
    rl.add_argument("--out", type=Path, required=True)
    rc = sub.add_parser("record", help="verify a frozen experiment record (artifact hashes, scorecard regeneration)")
    rc.add_argument("action", choices=["verify"])
    rc.add_argument("path", type=Path)
    sub.add_parser("probe", help="GPU, VRAM, telemetry and runtime checks (run on the GPU machine)")
    sv = sub.add_parser("serve", help="start vLLM on 127.0.0.1 with fixed, recorded settings (GPU machine)")
    sv.add_argument("--model", required=True)
    sv.add_argument("--revision", required=True)
    sv.add_argument("--lora", action="append", default=[], metavar="NAME=PATH")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--max-model-len", type=int, default=5120)
    sv.add_argument("--gpu-memory-utilization", type=float, default=0.87,
                    help="Windows keeps ~1.2 GiB of the 12 GiB card; 0.90 does not fit")
    sv.add_argument("--max-lora-rank", type=int, default=16)
    sv.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "splits":
        from modelforge import splits
        if args.action == "build":
            for path in splits.write(splits.build_finqa()):
                print(path)
        elif args.action == "build-v2":
            for path in splits.write([splits.build_v2()]):
                print(path)
        elif args.action == "build-comparison":
            for path in splits.write([splits.build_comparison()]):
                print(path)
        else:
            print(json.dumps(splits.verify(splits.load(args.name)), indent=1))
        return 0
    if args.command == "calc-suite":
        from modelforge.datasets import finqa
        print(json.dumps(_calc_suite(args.revision or finqa.REVISION), indent=1))
        return 0
    if args.command == "run":
        from modelforge import runner
        from modelforge.adapters import make_adapter
        options = {}
        if args.adapter == "qwen":
            if not args.model:
                parser.error("--model is required for the qwen adapter")
            options = {"model": args.model, "base_url": args.base_url, "label": args.label,
                       "max_tokens": args.max_tokens}
        adapter = make_adapter(args.adapter, **options)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out or ARTIFACTS / f"{stamp}-{adapter.config_id}-{args.selection}"
        print(runner.run(args.manifest, args.selection, adapter, out, args.limit, args.concurrency).read_text())
        return 0
    if args.command == "report":
        from modelforge import runner
        print(runner.write_scorecard(args.run_dir).read_text())
        return 0
    if args.command == "agree":
        from modelforge import runner
        print(json.dumps(runner.agreement(args.run_a, args.run_b), indent=1))
        return 0
    if args.command == "compare":
        from modelforge import runner
        print(json.dumps(runner.compare(args.run_a, args.run_b), indent=1))
        return 0
    if args.command == "diagnose":
        from modelforge import runner
        print(json.dumps(runner.diagnose(args.run_dir, args.baseline), indent=1))
        return 0
    if args.command == "relabel":
        from modelforge import runner
        print(runner.relabel(args.run_dir, args.labels, args.out).read_text())
        return 0
    if args.command == "record":
        from modelforge import records
        report = records.verify(args.path)
        print(json.dumps(report, indent=1))
        return 0 if report["passed"] else 1
    if args.command == "probe":
        from modelforge import probe
        return probe.main()
    if args.command == "serve":
        from modelforge import serve
        return serve.main(args.model, args.revision, serve.parse_loras(args.lora), dry_run=args.dry_run,
                          port=args.port, max_model_len=args.max_model_len,
                          gpu_memory_utilization=args.gpu_memory_utilization, max_lora_rank=args.max_lora_rank)
    return 2


if __name__ == "__main__":
    sys.exit(main())
