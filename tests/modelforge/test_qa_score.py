import json
from pathlib import Path

import pytest

from modelforge import qa_score as S


def test_explicit_gold_unit_scale_is_honoured_from_v1_7():
    gold = {"gold_value": "2018", "gold_unit": "USD millions", "decimals": 0, "sign_policy": "signed"}
    question = "What was adjusted EBITDA for FY 2023"
    assert S.numeric_outcome("2018000000", gold, question, version="v1.6") == "scale_slip"
    assert S.numeric_outcome("2018000000", gold, question, version="v1.7") == "correct"
    stated = {"gold_value": "1577", "gold_unit": "as stated in the question", "decimals": 0, "sign_policy": "signed"}
    assert S.numeric_outcome("1577000000", stated, "capex (in USD millions)?", version="v1.7") == "correct"


def test_unknown_company_raises_instead_of_calendar_fallback():
    assert S.fiscal_year_end(None) is None
    with pytest.raises(KeyError):
        S.fiscal_year_end("Not A Company Inc", {"3M": None})


def test_fiscal_files_merge_and_conflicts_raise(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"fiscal_year_end": {"NIKE": [5, 31]}}))
    b.write_text(json.dumps({"fiscal_year_end": {"NIKE": [6, 30]}}))
    assert S.load_fiscal([a]) == {"NIKE": (5, 31)}
    with pytest.raises(ValueError):
        S.load_fiscal([a, b])


@pytest.mark.skipif(not Path("artifacts/runs/20261110-qa15-fb-supplied-base").exists(), reason="run artifacts not present")
def test_v16_scoring_regenerates_the_frozen_dev_table():
    from modelforge import records
    report = records.verify(Path("evals/records/financebench-dev-trust-v17.json"))
    assert report["scores"] == "regenerated", report.get("score_tables")
