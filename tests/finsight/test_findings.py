from pathlib import Path

from finsight.contracts.report import ReportOutput
from finsight.findings import source_period, validate_report
from finsight.ingestion.html import parse_filing

PASSAGES = {p.id: p for p in parse_filing(Path("tests/fixtures/sec/synthetic-10k.html").read_text(), "sec:test", "mmm18")}
SALES = "mmm18:item_8:t0r2"
OPINC = "mmm18:item_8:t0r3"
CAPEX = "mmm18:item_8:t1r1"


def op(name, pid, text, value, period="FY2018", unit="USD", scale="millions"):
    return {"name": name, "passage_id": pid, "original_text": text, "value": value, "scale": scale, "unit": unit,
            "period": period}


def finding(metric, operands, program=(), status="answered", unit="USD", explanation=(), missing=None):
    return {"metric": metric, "period": "FY2018", "status": status, "operands": list(operands), "program": list(program),
            "unit": unit, "explanation": list(explanation), "missing": missing}


def run(*findings):
    return validate_report(ReportOutput.model_validate({"findings": list(findings), "unresolved": []}), PASSAGES)


def test_source_period_reads_the_column_header():
    assert source_period(PASSAGES[SALES], "32,765") == "2018" and source_period(PASSAGES[SALES], "31,657") == "2017"


def test_three_independent_findings_are_accepted():
    revenue, margin, capex = run(
        finding("revenue", [op("revenue_2018", SALES, "$ 32,765", "32765")]),
        finding("operating_margin", [op("operating_income", OPINC, "7,207", "7207"), op("revenue", SALES, "32,765", "32765")],
                [{"op": "divide", "args": ["operating_income", "revenue"]}], unit="ratio",
                explanation=[{"text": "Operating income was 7,207 on net sales of 32,765.", "passage_ids": [OPINC, SALES]}]),
        finding("capex", [op("capex", CAPEX, "(1,577)", "-1577")]))
    assert revenue.status == "accepted" and revenue.value == "32765000000"
    assert margin.status == "accepted" and margin.value.startswith("0.2199") and margin.claims[0].supported
    assert capex.status == "accepted" and capex.operands[0].source_period == "2018"


def test_wrong_year_column_fabricated_literal_uncited_passage_and_wrong_metric_are_rejected():
    wrong_year, fabricated, uncited, wrong_metric = run(
        finding("revenue", [op("revenue", SALES, "31,657", "31657")]),                     # 2017 column labelled FY2018
        finding("operating_margin", [op("operating_income", OPINC, "7,207", "7207"), op("revenue", SALES, "32,765", "32765")],
                [{"op": "divide", "args": ["operating_income", "32765"]}]),               # literal bypasses citation
        finding("revenue", [op("revenue", "mmm18:item_8:t9r9", "32,765", "32765")]),
        finding("revenue", [op("revenue", OPINC, "7,207", "7207")]))                       # operating income is not revenue
    assert wrong_year.status == "rejected" and "not 2018" in wrong_year.issues[0]
    assert fabricated.status == "rejected" and any("must be cited operands" in i for i in fabricated.issues)
    assert uncited.status == "rejected" and "not supplied" in uncited.issues[0]
    assert wrong_metric.status == "rejected" and any("no operand is revenue" in i for i in wrong_metric.issues)


def test_insufficient_evidence_is_its_own_finding():
    (missing,) = run(finding("net_income", [], status="insufficient_evidence", unit=None,
                             missing="the supplied passages do not report net income for FY2018"))
    assert missing.status == "insufficient_evidence" and "net income" in missing.issues[0]


def test_unit_comes_from_the_operands_not_the_model_label():
    # Seen on the real 3M filing: a model labelled revenue "ratio", which would render $32.8B as a percentage.
    revenue, margin = run(finding("revenue", [op("revenue", SALES, "$ 32,765", "32765")], unit="ratio"),
                          finding("operating_margin", [op("operating_income", OPINC, "7,207", "7207"),
                                                       op("revenue", SALES, "32,765", "32765")],
                                  [{"op": "divide", "args": ["operating_income", "revenue"]}], unit="USD"))
    assert revenue.unit == "USD" and margin.unit == "ratio"


def test_declaring_other_years_does_not_pass_an_answer_with_no_figure_from_the_requested_year():
    # Seen replaying wrong-period answers: both operands from other years, each declared, used to pass.
    growth = finding("revenue", [op("revenue_2018", SALES, "32,765", "32765", period="FY2018"),
                                 op("revenue_2017", SALES, "31,657", "31657", period="FY2017")],
                     [{"op": "subtract", "args": ["revenue_2018", "revenue_2017"]}])
    asked_absent_year = dict(growth, period="FY2016")
    ok, wrong = run(growth, asked_absent_year)
    assert ok.status == "accepted"                       # FY2018 present; the declared FY2017 operand is allowed
    assert wrong.status == "rejected" and any("not 2016" in i for i in wrong.issues)
