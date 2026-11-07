"""SEC EDGAR ingestion behind a source interface (direct HTTPS with httpx).

* HTTPS only, to www.sec.gov and data.sec.gov; redirects are followed manually and each hop's
  scheme and host is validated; responses have size limits and timeouts.
* Requests share a rate limit (default 5 per second) held in Postgres, so separate processes
  cannot exceed it together; an in-process limiter is used without a database.
* SEC requires a User-Agent with contact details (``FINSIGHT_SEC_USER_AGENT``); www.sec.gov
  refuses generic ones.
* Filing selection: the original 10-K whose report date is the requested fiscal year's end
  (from the company's fiscal-year-end), filed on or before the cutoff date. Amendments (10-K/A)
  are recorded separately and never chosen by default.
* Documents are stored with source URL, accession, retrieval time and sha256 and are served
  from the store on repeat requests. companyfacts responses aggregate facts across filings and
  change over time, so each fetch is kept as a timestamped snapshot.
* Facts are selected by concept, unit, exact reporting dates and the filing's own accession
  (as reported); the same period reported in later filings is returned separately as later
  comparatives or restatements.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

ALLOWED_HOSTS = {"www.sec.gov", "data.sec.gov"}
MAX_BYTES = {"document": 60_000_000, "json": 40_000_000}
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class SourceError(RuntimeError):
    pass


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise SourceError(f"refusing {url}: only https to {sorted(ALLOWED_HOSTS)} is allowed")


class LocalRateLimiter:
    def __init__(self, per_second: float = 5.0) -> None:
        self.interval = 1.0 / per_second
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_allowed)
            self.next_allowed = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


class PostgresRateLimiter:
    """Shared limit across processes: each request reserves the next slot in rate_limits."""

    def __init__(self, conn, per_second: float = 5.0) -> None:
        self.conn = conn
        self.interval_ms = int(1000 / per_second)

    def wait(self, host: str) -> None:
        row = self.conn.execute(
            "INSERT INTO rate_limits (host, min_interval_ms, next_allowed_at) VALUES (%s, %s, now() + make_interval(secs => %s)) "
            "ON CONFLICT (host) DO UPDATE SET next_allowed_at = greatest(rate_limits.next_allowed_at, now()) "
            "+ make_interval(secs => %s) RETURNING extract(epoch FROM (next_allowed_at - now())) - %s AS wait_s",
            (host, self.interval_ms, self.interval_ms / 1000, self.interval_ms / 1000, self.interval_ms / 1000)).fetchone()
        if row["wait_s"] > 0:
            time.sleep(float(row["wait_s"]))


@dataclass
class Filing:
    cik: str
    accession: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str

    @property
    def is_amendment(self) -> bool:
        return self.form.endswith("/A")

    @property
    def url(self) -> str:
        return (f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{self.accession.replace('-', '')}/"
                f"{self.primary_document}")


@dataclass
class FilingChoice:
    filing: Filing
    amendments: list[Filing] = field(default_factory=list)
    reason: str = ""


class SecSource:
    def __init__(self, user_agent: str | None, store: Path, *, rate_limiter=None, conn=None,
                 transport: httpx.BaseTransport | None = None) -> None:
        if not user_agent or "@" not in user_agent:
            raise SourceError("FINSIGHT_SEC_USER_AGENT must name you and give a contact email "
                              "(SEC refuses generic user agents)")
        self.store = store
        self.conn = conn
        self.limiter = rate_limiter or LocalRateLimiter()
        self.client = httpx.Client(headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
                                   timeout=TIMEOUT, follow_redirects=False, transport=transport)

    # ---------------------------------------------------------------- transport

    def get(self, url: str, kind: str = "json") -> tuple[bytes, str]:
        """GET with validated redirects and a size limit. Returns (body, final URL)."""
        for _ in range(4):
            _check_url(url)
            self.limiter.wait(urlsplit(url).hostname)
            with self.client.stream("GET", url) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                if response.status_code != 200:
                    raise SourceError(f"GET {url}: HTTP {response.status_code}")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES[kind]:
                        raise SourceError(f"GET {url}: response exceeds {MAX_BYTES[kind]} bytes")
                return bytes(body), url
        raise SourceError(f"GET {url}: too many redirects")

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url, "json")[0])

    # ---------------------------------------------------------------- discovery

    def resolve_cik(self, company: str) -> str:
        if company.isdigit():
            return company.zfill(10)
        tickers = self.get_json("https://www.sec.gov/files/company_tickers.json")
        wanted = company.strip().upper()
        for entry in tickers.values():
            if entry["ticker"].upper() == wanted or entry["title"].upper() == wanted:
                return str(entry["cik_str"]).zfill(10)
        raise SourceError(f"no SEC registrant matches {company!r}")

    def submissions(self, cik: str) -> dict:
        return self.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

    def filings(self, cik: str) -> tuple[list[Filing], str | None]:
        """All filings (recent plus older pages) and the fiscal-year end as MMDD."""
        data = self.submissions(cik)
        pages = [data["filings"]["recent"]]
        for extra in data["filings"].get("files", []):
            pages.append(self.get_json(f"https://data.sec.gov/submissions/{extra['name']}"))
        out = []
        for page in pages:
            for i, form in enumerate(page["form"]):
                report = page.get("reportDate", [""] * len(page["form"]))[i]
                out.append(Filing(cik=cik, accession=page["accessionNumber"][i], form=form,
                                  filing_date=date.fromisoformat(page["filingDate"][i]),
                                  report_date=date.fromisoformat(report) if report else None,
                                  primary_document=page["primaryDocument"][i]))
        return out, data.get("fiscalYearEnd")

    @staticmethod
    def choose_annual(filings: list[Filing], fiscal_year: int, fiscal_year_end: str | None,
                      cutoff: date | None = None) -> FilingChoice:
        month, day = (int(fiscal_year_end[:2]), int(fiscal_year_end[2:])) if fiscal_year_end else (12, 31)
        expected_end = date(fiscal_year, month, day)
        eligible = [f for f in filings if cutoff is None or f.filing_date <= cutoff]
        matches = [f for f in eligible if f.form in ("10-K", "10-K/A") and f.report_date is not None
                   and abs((f.report_date - expected_end).days) <= 7]   # 52/53-week years end near the date
        originals = sorted((f for f in matches if not f.is_amendment), key=lambda f: f.filing_date)
        if not originals:
            raise SourceError(f"no 10-K for fiscal year {fiscal_year} (period end ~{expected_end})"
                              + (f" filed by {cutoff}" if cutoff else ""))
        return FilingChoice(filing=originals[0], amendments=[f for f in matches if f.is_amendment],
                            reason=f"original 10-K with report date {originals[0].report_date} for FY{fiscal_year}")

    # ---------------------------------------------------------------- storage

    def fetch_document(self, filing: Filing) -> dict:
        """Download (or reuse) the filing's primary document; returns the stored document record."""
        if self.conn is not None:
            existing = self.conn.execute("SELECT * FROM documents WHERE source_uri = %s ORDER BY retrieved_at LIMIT 1",
                                         (filing.url,)).fetchone()
            if existing and Path(existing["artifact_path"]).exists():
                return existing
        body, final_url = self.get(filing.url, "document")
        digest = hashlib.sha256(body).hexdigest()
        directory = self.store / "sec" / filing.cik / filing.accession
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filing.primary_document
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        record = {"id": f"sec:{filing.accession}:{filing.primary_document}", "source_uri": filing.url,
                  "sha256": digest, "artifact_path": str(path), "accession": filing.accession, "cik": filing.cik,
                  "form": filing.form, "fiscal_year": filing.report_date.year if filing.report_date else None,
                  "period_end": filing.report_date, "filing_date": filing.filing_date, "content_type": "text/html",
                  "size_bytes": len(body), "retrieved_at": datetime.now(timezone.utc), "company": None,
                  "parser_version": None}
        if self.conn is not None:
            columns = ", ".join(record)
            self.conn.execute(f"INSERT INTO documents ({columns}) VALUES ({', '.join(['%s'] * len(record))}) "
                              "ON CONFLICT DO NOTHING", tuple(record.values()))
        return record

    def company_facts_snapshot(self, cik: str) -> tuple[dict, dict]:
        """Fetch companyfacts and keep it as a timestamped snapshot. Returns (facts, snapshot record)."""
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        body, _ = self.get(url, "json")
        retrieved = datetime.now(timezone.utc)
        directory = self.store / "sec" / cik / "companyfacts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{retrieved.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        path.write_bytes(body)
        snapshot = {"kind": "sec_companyfacts", "source_uri": url, "sha256": hashlib.sha256(body).hexdigest(),
                    "artifact_path": str(path), "retrieved_at": retrieved}
        if self.conn is not None:
            self.conn.execute("INSERT INTO source_snapshots (kind, source_uri, sha256, artifact_path, retrieved_at) "
                              "VALUES (%(kind)s, %(source_uri)s, %(sha256)s, %(artifact_path)s, %(retrieved_at)s)", snapshot)
        return json.loads(body), snapshot


def select_facts(facts: dict, concept: str, unit: str, accession: str, start: date | None, end: date,
                 taxonomy: str = "us-gaap") -> dict:
    """Exact fact selection for one filing: as reported in that filing, plus the same period in later filings."""
    entries = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
    same_period = [e for e in entries if e.get("end") == end.isoformat()
                   and (start is None or e.get("start") == start.isoformat())]
    as_reported = [e for e in same_period if e.get("accn") == accession]
    later = sorted(({k: e[k] for k in ("accn", "val", "form", "filed", "fy", "fp")} for e in same_period
                    if e.get("accn") != accession), key=lambda e: e["filed"])
    values = {e["val"] for e in as_reported}
    if len(values) > 1:
        raise SourceError(f"{concept}: filing {accession} reports conflicting values {sorted(values)} for {start}..{end}")
    return {"concept": concept, "unit": unit, "accession": accession, "start": start, "end": end,
            "as_reported": as_reported[0]["val"] if as_reported else None,
            "later_filings": later,
            "restated": sorted({e["val"] for e in later} - values) if as_reported else []}
