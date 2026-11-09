"""FinSight command line.

    finsight brief --company MMM [--commodity wti --commodity copper] --out DIR
        writes DIR/pack.json (every figure with its source and as-of date) and DIR/draft.md (deterministic report)
    finsight check-report DIR/report.md [--pack DIR/pack.json]
        exits 1 if the report contains a number that is not in the pack
    finsight commodities
        lists the market series available

Runs without Postgres, Docker or a GPU (file cache, local SEC rate limit). Needs FINSIGHT_SEC_USER_AGENT for company
data and FINSIGHT_FRED_API_KEY for market data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from finsight.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finsight")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("brief", help="build a context pack and a draft report")
    b.add_argument("--company", help="ticker or CIK")
    b.add_argument("--commodity", action="append", default=[], help="series key (see `finsight commodities`)")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check-report", help="fail if a report has numbers not in the pack")
    c.add_argument("report", type=Path)
    c.add_argument("--pack", type=Path, default=None)
    sub.add_parser("commodities", help="list market series")
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "commodities":
        from finsight.market import CATALOG
        for key, (series_id, title, unit) in CATALOG.items():
            print(f"{key:9} {series_id:13} {title} ({unit})")
        return 0
    if args.command == "check-report":
        from finsight.insights import check_report
        pack = json.loads((args.pack or args.report.parent / "pack.json").read_text())
        bad = check_report(args.report.read_text(), pack)
        if bad:
            print("numbers not found in the pack: " + ", ".join(bad), file=sys.stderr)
            return 1
        print("every number in the report is in the pack")
        return 0

    from finsight.insights import build_pack, company_pack, markdown
    if not args.company and not args.commodity:
        parser.error("give --company and/or --commodity")
    store = settings.state_dir / "artifacts"
    company = None
    if args.company:
        from finsight.ingestion.sec import LocalRateLimiter, SecSource
        company = company_pack(SecSource(settings.sec_user_agent, store, rate_limiter=LocalRateLimiter()), args.company)
    commodities, problems = [], []
    if args.commodity:
        from finsight.market import FredSource, MarketError, analytics
        try:
            fred = FredSource(settings.fred_api_key.get_secret_value() if settings.fred_api_key else None, store)
            commodities = [analytics(fred.series(key)) for key in args.commodity]
        except MarketError as exc:
            problems.append(str(exc))
    pack = build_pack(company, commodities)
    if problems:
        pack["problems"] = problems
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pack.json").write_text(json.dumps(pack, indent=1, default=str) + "\n")
    (args.out / "draft.md").write_text(markdown(pack))
    print(json.dumps({"pack": str(args.out / "pack.json"), "draft": str(args.out / "draft.md"),
                      "insights": len(pack["insights"]), "problems": problems}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
