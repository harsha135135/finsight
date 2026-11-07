import json
from pathlib import Path

from modelforge.datasets.financebench import REVISION, split_companies


def test_company_split_is_seeded_balanced_and_keeps_3m_in_development():
    questions = [{"company": c, "question_type": "t"} for c, n in
                 {"3M": 8, "A": 9, "B": 7, "C": 5, "D": 5, "E": 3, "F": 2, "G": 1}.items() for _ in range(n)]
    first, second = split_companies(questions), split_companies(questions)
    assert first == second and "3M" in first["dev"]
    assert not set(first["dev"]) & set(first["held_out"])
    sizes = {s: sum(q["company"] in cs for q in questions) for s, cs in first.items()}
    assert abs(sizes["dev"] - sizes["held_out"]) <= 2


def test_committed_manifest_separates_companies_and_covers_every_document():
    m = json.loads(Path("evals/manifests/financebench-v1.json").read_text())
    assert m["revision"] == REVISION and m["license"].startswith("CC-BY-NC")
    assert not set(m["companies"]["dev"]) & set(m["companies"]["held_out"]) and "3M" in m["companies"]["dev"]
    assert len(m["samples"]["dev"]) + len(m["samples"]["held_out"]) == 150
    assert all(d["category"] == "original" for d in m["documents"].values()) and len(m["documents"]) == 84
