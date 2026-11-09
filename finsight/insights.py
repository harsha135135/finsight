"""Insights engine: one context pack for a company and/or commodities, used by the dashboard, the API and the CLI.

Company data comes from SEC EDGAR (submissions and XBRL companyfacts); market data from FRED. Every insight is a
templated sentence that carries its figures, sources (accession or series id) and as-of date; nothing is generated
by a model here. Company-commodity links are shown as co-movement series only, never as causal statements, and
nothing here is investment advice.

Quarters: only ~90-day periods count as quarters (10-Qs also report year-to-date periods); a fourth quarter is the
full year minus the nine-month year-to-date figure and is labelled derived. Year-over-year changes compare a figure
with the comparative column of the same filing, so both sides are on the same basis.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from finsight.retrieval.search import METRICS

FLOW_METRICS = {"revenue": METRICS["revenue"]["concepts"], "operating_income": METRICS["operating_income"]["concepts"],
                "net_income": METRICS["net_income"]["concepts"]}
CAPEX = METRICS["capex"]["concepts"]


def money(v: float) -> str:
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f} bn"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.0f} m"
    return f"{sign}${a:,.0f}"


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%"


def _days(e: dict) -> int | None:
    if not e.get("start"):
        return None
    return (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days


def _entries(facts: dict, concepts: list[str]) -> tuple[str | None, list[dict]]:
    for concept in concepts:
        units = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {})
        entries = [e for e in units.get("USD", []) if e.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A")]
        if entries:
            return concept, entries
    return None, []


def _first_filed(entries: list[dict]) -> dict[tuple[str, str], dict]:
    """As originally reported: the earliest filed entry for each (start, end)."""
    out: dict = {}
    for e in sorted(entries, key=lambda e: e["filed"]):
        out.setdefault((e["start"], e["end"]), e)
    return out


def quarterly(entries: list[dict]) -> list[dict]:
    flows = [e for e in entries if e.get("start")]
    original = _first_filed(flows)
    quarters = {k: e for k, e in original.items() if 80 <= _days(e) <= 100}
    annual = {k: e for k, e in original.items() if 350 <= _days(e) <= 380}
    ytd9 = {k: e for k, e in original.items() if 260 <= _days(e) <= 285}
    points = [{"end": e["end"], "value": e["val"], "accession": e["accn"], "form": e["form"], "filed": e["filed"], "derived": False}
              for e in quarters.values()]
    have = {p["end"] for p in points}
    for (start, end), year in annual.items():
        if end in have:
            continue
        nine = next((e for (s, _), e in ytd9.items() if s == start), None)
        if nine:
            points.append({"end": end, "value": year["val"] - nine["val"], "accession": year["accn"], "form": "10-K",
                           "filed": year["filed"], "derived": True, "derivation": f"full year {year['accn']} minus nine months {nine['accn']}"})
    return sorted(points, key=lambda p: p["end"])


def same_filing_comparative(entries: list[dict], current: dict) -> dict | None:
    """The prior-year figure for the same period length reported in the same filing as ``current``."""
    end, days = date.fromisoformat(current["end"]), _days(current)
    for e in entries:
        if e["accn"] != current["accn"] or not e.get("start"):
            continue
        gap = (end - date.fromisoformat(e["end"])).days
        if 355 <= gap <= 375 and abs(_days(e) - days) <= 10:
            return e
    return None


def company_pack(source, ticker: str, *, today: date | None = None) -> dict:
    today = today or date.today()
    cik = source.resolve_cik(ticker)
    submissions = source.submissions(cik)
    facts, snapshot = source.company_facts_snapshot(cik)
    recent = submissions["filings"]["recent"]
    filings = [{"form": recent["form"][i], "filed": recent["filingDate"][i], "period": recent["reportDate"][i],
                "accession": recent["accessionNumber"][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{recent['accessionNumber'][i].replace('-', '')}/{recent['primaryDocument'][i]}"}
               for i in range(len(recent["form"])) if recent["form"][i] in ("10-K", "10-Q", "8-K")][:10]
    metrics, insights = {}, []
    for name, concepts in FLOW_METRICS.items():
        concept, entries = _entries(facts, concepts)
        if not entries:
            continue
        q = quarterly(entries)
        latest_raw = max((e for e in entries if e.get("start") and 80 <= _days(e) <= 100), key=lambda e: (e["end"], -_ts(e["filed"])), default=None)
        if latest_raw:
            latest_raw = _first_filed([e for e in entries if e.get("start") and e["end"] == latest_raw["end"] and 80 <= _days(e) <= 100])[(latest_raw["start"], latest_raw["end"])]
        comp = same_filing_comparative(entries, latest_raw) if latest_raw else None
        ttm = sum(p["value"] for p in q[-4:]) if len(q) >= 4 else None
        metrics[name] = {"concept": concept, "quarters": q[-12:], "ttm": ttm, "ttm_quarters": [p["end"] for p in q[-4:]],
                         "latest": latest_raw and {"end": latest_raw["end"], "value": latest_raw["val"], "accession": latest_raw["accn"],
                                                   "form": latest_raw["form"], "filed": latest_raw["filed"]},
                         "year_ago_same_filing": comp and {"end": comp["end"], "value": comp["val"]}}
    rev, oi = metrics.get("revenue"), metrics.get("operating_income")
    if rev and rev["latest"]:
        l, c = rev["latest"], rev["year_ago_same_filing"]
        text = f"Revenue for the quarter ended {l['end']} was {money(l['value'])}"
        figures = [money(l["value"])]
        if c and c["value"]:
            change = (l["value"] - c["value"]) / abs(c["value"])
            text += f", {pct(change)} from {money(c['value'])} a year earlier"
            figures += [pct(change), money(c["value"])]
        insights.append(_insight(text + f" ({l['form']} {l['accession']}).", figures, [_sec(l, cik)], l["end"]))
        if rev["ttm"]:
            insights.append(_insight(f"Trailing-twelve-month revenue (quarters ending {rev['ttm_quarters'][0]} to {rev['ttm_quarters'][-1]}"
                                     f"{', including a derived fourth quarter' if any(p['derived'] for p in rev['quarters'][-4:]) else ''}) was {money(rev['ttm'])}.",
                                     [money(rev["ttm"])], [_sec(rev["latest"], cik)], rev["ttm_quarters"][-1]))
    if rev and oi and rev["latest"] and oi["latest"] and rev["latest"]["end"] == oi["latest"]["end"] and rev["latest"]["value"]:
        margin = oi["latest"]["value"] / rev["latest"]["value"]
        text, figures = f"Operating margin for the quarter ended {oi['latest']['end']} was {margin * 100:.1f}%", [f"{margin * 100:.1f}%"]
        if rev["year_ago_same_filing"] and oi["year_ago_same_filing"] and rev["year_ago_same_filing"]["value"]:
            prior = oi["year_ago_same_filing"]["value"] / rev["year_ago_same_filing"]["value"]
            text += f", versus {prior * 100:.1f}% a year earlier in the same filing"
            figures.append(f"{prior * 100:.1f}%")
        insights.append(_insight(text + ".", figures, [_sec(oi["latest"], cik)], oi["latest"]["end"]))
    concept, capex_entries = _entries(facts, CAPEX)
    if capex_entries:
        flows = [e for e in capex_entries if e.get("start")]
        latest = max(flows, key=lambda e: (e["end"], _days(e)))
        comp = same_filing_comparative(capex_entries, latest)
        metrics["capex"] = {"concept": concept, "latest_period": {"start": latest["start"], "end": latest["end"], "value": latest["val"],
                                                                  "accession": latest["accn"], "form": latest["form"], "days": _days(latest)},
                            "year_ago_same_filing": comp and {"end": comp["end"], "value": comp["val"]}}
        label = "the year" if _days(latest) >= 350 else f"the {round(_days(latest) / 30.4)} months"
        text, figures = f"Capital expenditure for {label} ended {latest['end']} was {money(latest['val'])}", [money(latest["val"])]
        if comp and comp["val"]:
            text += f", {pct((latest['val'] - comp['val']) / abs(comp['val']))} from {money(comp['val'])} a year earlier"
            figures += [pct((latest["val"] - comp["val"]) / abs(comp["val"])), money(comp["val"])]
        insights.append(_insight(text + f" ({latest['form']} {latest['accn']}).", figures, [_sec({"accession": latest["accn"], "form": latest["form"], "filed": latest["filed"]}, cik)], latest["end"]))
    if filings:
        f = filings[0]
        insights.append(_insight(f"Most recent filing: {f['form']} filed {f['filed']}.", [], [{"type": "sec", "accession": f["accession"], "form": f["form"], "url": f["url"]}], f["filed"]))
    return {"ticker": ticker.upper(), "cik": cik, "name": submissions.get("name"), "industry": submissions.get("sicDescription"),
            "fiscal_year_end": submissions.get("fiscalYearEnd"), "filings": filings, "metrics": metrics, "insights": insights,
            "sources": {"companyfacts_snapshot": {k: str(v) for k, v in snapshot.items() if k != "artifact_path"}},
            "basis": "SEC XBRL as originally reported; quarters are ~90-day periods, a fourth quarter may be derived (full year minus nine months)"}


def commodity_insight(a: dict) -> dict:
    unit = a["unit"]
    shown = lambda v: f"{v:,.2f}"
    text = f"{a['title']} was {shown(a['latest']['value'])} ({unit}) on {a['latest']['date']}"
    figures = [shown(a["latest"]["value"])]
    for label, key in (("1 month", "change_1m"), ("1 year", "change_1y")):
        if a.get(key):
            text += f"; {pct(a[key]['pct'])} over {label}"
            figures.append(pct(a[key]["pct"]))
    r = a["range_52w"]
    if r["position"] is not None:
        text += f"; within its 52-week range of {shown(r['low'])}-{shown(r['high'])} it sits at {r['position'] * 100:.0f}%"
        figures += [shown(r["low"]), shown(r["high"]), f"{r['position'] * 100:.0f}%"]
    if a.get("moving_averages"):
        text += f"; the latest value is {a['moving_averages']['trend']} its 50- and 200-day averages"
    text += f" ({a['freshness']})."
    return _insight(text, figures, [{"type": "fred", "series_id": a["series_id"], "url": a["source_url"], "retrieved_at": a["retrieved_at"]}], a["latest"]["date"])


def build_pack(company: dict | None, commodities: list[dict]) -> dict:
    insights = (company or {}).get("insights", []) + [commodity_insight(a) for a in commodities if a.get("available")]
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "company": company, "commodities": commodities,
            "insights": insights,
            "rules": ["every figure in a report must come from this pack", "state each figure's as-of date; monthly series lag",
                      "company-commodity co-movement is not causation", "no investment advice"]}


def markdown(pack: dict) -> str:
    c = pack.get("company")
    lines = [f"# {c['name']} ({c['ticker']})" if c else "# Market brief", "",
             f"Generated {pack['generated_at']}. Figures as originally reported in SEC filings and as published by FRED; "
             "each line gives its source and as-of date. Not investment advice.", ""]
    if c:
        lines += [f"Industry: {c.get('industry') or 'n/a'}; fiscal year ends {c.get('fiscal_year_end') or 'n/a'}.", "", "## Company", ""]
        lines += [f"- {i['text']}" for i in c["insights"]]
    if pack["commodities"]:
        lines += ["", "## Markets", ""]
        lines += [f"- {commodity_insight(a)['text']}" for a in pack["commodities"] if a.get("available")]
    sources = [f"- {s.get('form') or s.get('series_id', '')} {s.get('accession', '')} {s.get('url', '')}".replace("  ", " ").rstrip()
               for i in pack["insights"] for s in i["sources"]]
    lines += ["", "## Sources", *dict.fromkeys(sources)]
    return "\n".join(lines) + "\n"


_NUMBER = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        plain = raw.replace("$", "").replace(",", "").lstrip("+")
        out.add(plain.lstrip("-"))
    return out


def check_report(report: str, pack: dict) -> list[str]:
    """Numbers in a report that do not appear in the pack's insight texts or figures (years and dates excepted)."""
    allowed: set[str] = set()
    for i in pack.get("insights", []):
        allowed |= _numbers(i["text"]) | {n for f in i.get("figures", []) for n in _numbers(f)}
    unsupported = []
    for n in sorted(_numbers(re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", report))):
        if re.fullmatch(r"(19|20)\d{2}", n) or n in allowed or (n.isdigit() and int(n) <= 12):
            continue
        unsupported.append(n)
    return unsupported


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _sec(entry: dict, cik: str) -> dict:
    return {"type": "sec", "accession": entry["accession"], "form": entry.get("form"), "filed": entry.get("filed"),
            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={entry.get('form', '')}"}


def _insight(text: str, figures: list[str], sources: list[dict], as_of: str) -> dict:
    return {"text": text, "figures": figures, "sources": sources, "as_of": as_of}
