import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from finsight.insights import build_pack, check_report, commodity_insight, quarterly, same_filing_comparative
from finsight.market import FredSource, MarketError, Series, analytics


def e(start, end, val, accn, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "accn": accn, "filed": filed, "form": form}


ENTRIES = [
    e("2023-01-01", "2023-03-31", 100, "q1-23", "2023-04-25"), e("2023-04-01", "2023-06-30", 110, "q2-23", "2023-07-25"),
    e("2023-01-01", "2023-06-30", 210, "q2-23", "2023-07-25"),                       # year-to-date, not a quarter
    e("2023-07-01", "2023-09-30", 120, "q3-23", "2023-10-25"), e("2023-01-01", "2023-09-30", 330, "q3-23", "2023-10-25"),
    e("2023-01-01", "2023-12-31", 460, "k-23", "2024-02-05", "10-K"),
    e("2024-01-01", "2024-03-31", 115, "q1-24", "2024-04-25"), e("2023-01-01", "2023-03-31", 100, "q1-24", "2024-04-25"),
]


def test_quarters_exclude_year_to_date_and_derive_the_fourth_quarter():
    q = quarterly(ENTRIES)
    assert [p["end"] for p in q] == ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]
    q4 = q[3]
    assert q4["derived"] and q4["value"] == 130 and "minus nine months" in q4["derivation"]


def test_year_over_year_uses_the_comparative_in_the_same_filing():
    latest = ENTRIES[6]
    comp = same_filing_comparative(ENTRIES, latest)
    assert comp["accn"] == "q1-24" and comp["val"] == 100


def _series(values, start=date(2024, 1, 1), frequency="Daily"):
    obs = [(start + timedelta(days=i), v) for i, v in enumerate(values)]
    return Series("wti", "DCOILWTICO", "WTI crude oil", "USD per barrel", frequency, obs, datetime.now(timezone.utc), "https://api.stlouisfed.org/fred/series/observations?series_id=DCOILWTICO")


def test_analytics_changes_range_trend_and_freshness():
    values = [50 + i * 0.1 for i in range(400)]
    a = analytics(_series(values), today=date(2024, 1, 1) + timedelta(days=402))
    assert a["latest"]["value"] == pytest.approx(89.9) and a["age_days"] == 3
    assert a["change_1y"]["pct"] == pytest.approx((89.9 - 53.4) / 53.4, rel=1e-3)
    assert a["range_52w"]["position"] == pytest.approx(1.0) and a["moving_averages"]["trend"] == "above both"
    assert a["freshness"].startswith("daily")
    monthly = analytics(_series([100, 101, 103], frequency="Monthly"), today=date(2024, 3, 1))
    assert "published with a lag" in monthly["freshness"] and "moving_averages" not in monthly
    text = commodity_insight(a)["text"]
    assert "WTI crude oil was 89.90" in text and "over 1 year" in text


def test_fred_client_never_leaks_the_key_and_caches(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path.endswith("/observations"):
            return httpx.Response(200, json={"observations": [{"date": "2026-11-06", "value": "61.2"}, {"date": "2026-11-07", "value": "."}]})
        return httpx.Response(200, json={"seriess": [{"frequency": "Daily", "title": "WTI", "last_updated": "x"}]})
    source = FredSource("SECRET-KEY", tmp_path, transport=httpx.MockTransport(handler))
    s = source.series("wti")
    assert s.observations == [(date(2026, 11, 6), 61.2)] and "SECRET-KEY" not in s.source_url
    assert not any("SECRET-KEY" in p.read_text() for p in tmp_path.rglob("*.json"))
    source.series("wti")
    assert len(calls) == 2                                                     # second call served from the snapshot
    failing = FredSource("SECRET-KEY", tmp_path / "x", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(MarketError) as err:
        failing.series("wti")
    assert "SECRET-KEY" not in str(err.value)
    with pytest.raises(MarketError):
        FredSource(None, tmp_path)


def test_check_report_rejects_numbers_not_in_the_pack():
    a = analytics(_series([50 + i * 0.1 for i in range(400)]), today=date(2025, 2, 10))
    pack = build_pack(None, [a])
    ok = f"WTI was 89.90 dollars on {a['latest']['date']} in 2025; up {pack['insights'][0]['figures'][1]} over a year."
    assert check_report(ok, pack) == []
    assert check_report("WTI will reach 120.00 next year.", pack) == ["120.00"]
