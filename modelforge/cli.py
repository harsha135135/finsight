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
    sp.add_argument("action", choices=["build", "verify"])
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
    rp.add_argument("--out", type=Path)
    rep = sub.add_parser("report", help="regenerate a scorecard from stored predictions")
    rep.add_argument("run_dir", type=Path)
    sub.add_parser("probe", help="GPU, VRAM, telemetry and runtime checks (run on the GPU machine)")
    args = parser.parse_args(argv)

    if args.command == "splits":
        from modelforge import splits
        if args.action == "build":
            for path in splits.write(splits.build_finqa()):
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
            options = {"model": args.model, "base_url": args.base_url, "label": args.label}
        adapter = make_adapter(args.adapter, **options)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.out or ARTIFACTS / f"{stamp}-{adapter.config_id}-{args.selection}"
        print(runner.run(args.manifest, args.selection, adapter, out, args.limit).read_text())
        return 0
    if args.command == "report":
        from modelforge import runner
        print(runner.write_scorecard(args.run_dir).read_text())
        return 0
    if args.command == "probe":
        from modelforge import probe
        return probe.main()
    return 2


if __name__ == "__main__":
    sys.exit(main())
