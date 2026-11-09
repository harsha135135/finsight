"""Market and macro data from FRED (Federal Reserve Bank of St. Louis) through its official API.

* The API key comes from FINSIGHT_FRED_API_KEY and never appears in logs, errors or stored files.
* HTTPS to api.stlouisfed.org only, size limits and timeouts; a generic client name (no personal contact).
* Each fetch is kept as a timestamped snapshot; repeat requests within ``max_age`` reuse the latest snapshot.
* Frequencies differ: daily series (oil, gas, yields, dollar index) and monthly series (metals, grains, CPI)
  that are published with a lag of weeks. Every result states its latest observation date and frequency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

API = "https://api.stlouisfed.org/fred"
USER_AGENT = "FinSight/0.1 (financial research tool)"
MAX_BYTES = 20_000_000

CATALOG = {
    "wti": ("DCOILWTICO", "WTI crude oil", "USD per barrel"),
    "brent": ("DCOILBRENTEU", "Brent crude oil", "USD per barrel"),
    "natgas": ("DHHNGSP", "Henry Hub natural gas", "USD per MMBtu"),
    "copper": ("PCOPPUSDM", "Copper (global price)", "USD per metric ton"),
    "aluminum": ("PALUMUSDM", "Aluminum (global price)", "USD per metric ton"),
    "wheat": ("PWHEAMTUSDM", "Wheat (global price)", "USD per metric ton"),
    "corn": ("PMAIZMTUSDM", "Corn (global price)", "USD per metric ton"),
    "ust10y": ("DGS10", "10-year US Treasury yield", "percent"),
    "cpi": ("CPIAUCSL", "US consumer prices (CPI-U)", "index 1982-84=100"),
    "usd": ("DTWEXBGS", "Broad US dollar index", "index Jan 2006=100"),
}


class MarketError(RuntimeError):
    pass


@dataclass
class Series:
    key: str
    series_id: str
    title: str
    unit: str
    frequency: str
    observations: list[tuple[date, float]]
    retrieved_at: datetime
    source_url: str          # without the API key
    notes: str = ""


class FredSource:
    def __init__(self, api_key: str | None, store: Path, *, transport: httpx.BaseTransport | None = None,
                 max_age: timedelta = timedelta(hours=6)) -> None:
        if not api_key:
            raise MarketError("FINSIGHT_FRED_API_KEY is not set (free key: fred.stlouisfed.org, My Account > API Keys)")
        self._key = api_key
        self.store = store / "fred"
        self.max_age = max_age
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(30.0, connect=10.0),
                                   follow_redirects=False, transport=transport)

    def _get(self, path: str, params: dict) -> dict:
        url = f"{API}/{path}"
        try:
            response = self.client.get(url, params={**params, "api_key": self._key, "file_type": "json"})
        except httpx.HTTPError as exc:
            raise MarketError(f"FRED {path}: {type(exc).__name__}") from None   # never echo the URL (it holds the key)
        if response.status_code != 200:
            raise MarketError(f"FRED {path} {params.get('series_id')}: HTTP {response.status_code}")
        if len(response.content) > MAX_BYTES:
            raise MarketError(f"FRED {path}: response too large")
        return response.json()

    def series(self, key: str, years: int = 6) -> Series:
        if key not in CATALOG:
            raise MarketError(f"unknown series {key!r}; known: {sorted(CATALOG)}")
        series_id, title, unit = CATALOG[key]
        cached = sorted((self.store / series_id).glob("*.json")) if (self.store / series_id).exists() else []
        if cached:
            data = json.loads(cached[-1].read_text())
            if datetime.now(timezone.utc) - datetime.fromisoformat(data["retrieved_at"]) < self.max_age:
                return _series(key, data)
        start = (date.today() - timedelta(days=365 * years)).isoformat()
        info = self._get("series", {"series_id": series_id})["seriess"][0]
        obs = self._get("series/observations", {"series_id": series_id, "observation_start": start})["observations"]
        retrieved = datetime.now(timezone.utc)
        data = {"key": key, "series_id": series_id, "title": title, "unit": unit, "frequency": info.get("frequency", ""),
                "fred_title": info.get("title"), "last_updated": info.get("last_updated"), "notes": (info.get("notes") or "")[:600],
                "observations": [[o["date"], o["value"]] for o in obs], "retrieved_at": retrieved.isoformat(),
                "source_url": f"{API}/series/observations?series_id={series_id}"}
        body = json.dumps(data).encode()
        path = self.store / series_id / f"{retrieved.strftime('%Y%m%dT%H%M%S')}-{hashlib.sha256(body).hexdigest()[:12]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return _series(key, data)


def _series(key: str, data: dict) -> Series:
    observations = [(date.fromisoformat(d), float(v)) for d, v in data["observations"] if v not in (".", "", None)]
    return Series(key=key, series_id=data["series_id"], title=data["title"], unit=data["unit"], frequency=data["frequency"],
                  observations=observations, retrieved_at=datetime.fromisoformat(data["retrieved_at"]),
                  source_url=data["source_url"], notes=data.get("notes", ""))


# ----------------------------------------------------------------------------- analytics (pure)

def value_on_or_before(observations: list[tuple[date, float]], when: date) -> tuple[date, float] | None:
    earlier = [o for o in observations if o[0] <= when]
    return earlier[-1] if earlier else None


def analytics(series: Series, today: date | None = None) -> dict:
    obs = series.observations
    if not obs:
        return {"key": series.key, "available": False}
    today = today or date.today()
    last_date, last = obs[-1]
    daily = "daily" in series.frequency.lower()

    def change(days: int):
        base = value_on_or_before(obs, last_date - timedelta(days=days))
        if base is None or base[1] == 0:
            return None
        return {"from_date": base[0].isoformat(), "from": base[1], "pct": (last - base[1]) / abs(base[1])}
    window = [v for d, v in obs if d > last_date - timedelta(days=365)]
    high, low = max(window), min(window)
    out = {"key": series.key, "series_id": series.series_id, "title": series.title, "unit": series.unit,
           "frequency": series.frequency, "available": True, "latest": {"date": last_date.isoformat(), "value": last},
           "age_days": (today - last_date).days, "change_1m": change(30), "change_3m": change(91), "change_1y": change(365),
           "range_52w": {"low": low, "high": high, "position": (last - low) / (high - low) if high > low else None},
           "retrieved_at": series.retrieved_at.isoformat(), "source_url": series.source_url}
    if daily and len(obs) >= 200:
        values = [v for _, v in obs]
        ma50, ma200 = sum(values[-50:]) / 50, sum(values[-200:]) / 200
        out["moving_averages"] = {"ma50": ma50, "ma200": ma200,
                                  "trend": "above both" if last > ma50 and last > ma200 else "below both" if last < ma50 and last < ma200 else "between"}
        returns = [math.log(values[i] / values[i - 1]) for i in range(len(values) - 63, len(values)) if values[i - 1] > 0 and values[i] > 0]
        if len(returns) > 20:
            mean = sum(returns) / len(returns)
            out["volatility_3m_annualized"] = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)) * math.sqrt(252)
    out["freshness"] = (f"daily; latest observation {last_date.isoformat()}" if daily else
                        f"{series.frequency.lower() or 'periodic'}; latest observation {last_date.isoformat()} (published with a lag)")
    return out
