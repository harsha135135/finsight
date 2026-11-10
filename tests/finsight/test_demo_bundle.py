import json
from pathlib import Path

import pytest

from finsight.demo_bundle import LOCAL_ENDPOINT, _scrub, problems


def test_scrub_removes_worker_identity_endpoints_and_absolute_paths():
    raw = {"model_config": {"base_url": "http://host.docker.internal:8000", "model": "m"},
           "detail": {"worker": "142dc82d098c:7"},
           "report": {"json_path": "/app/.finsight/artifacts/reports/x/report.json"},
           "documents": [{"artifact_path": "/Users/someone/dev/.finsight/artifacts/sec/1/a.htm"}]}
    out = _scrub(raw)
    assert out["model_config"]["base_url"] == LOCAL_ENDPOINT
    assert out["detail"]["worker"] == "<worker>"
    assert out["report"]["json_path"] == ".finsight/artifacts/reports/x/report.json"
    assert out["documents"][0]["artifact_path"] == ".finsight/artifacts/sec/1/a.htm"
    assert problems(json.dumps(out)) == []


@pytest.mark.parametrize("text", ["contact a.b@example.com", "key sk-ant-abcdefghijk", "host 192.168.0.128",
                                  "path /Users/someone/x", "http://127.0.0.1:8000/v1"])
def test_problems_flags_sensitive_strings(text):
    assert problems(text)


def test_committed_bundles_are_clean():
    for path in Path("evals/demo").glob("*/bundle.json"):
        assert problems(path.read_text()) == [], path
