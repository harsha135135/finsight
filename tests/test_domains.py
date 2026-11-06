import json
import unittest

from aiagents.domains import calculate_metrics, domain_instructions


class DomainTests(unittest.TestCase):
    def source(self, periods, **overrides):
        return {"id": "financials", "data": {"currency": "USD", "scale": "millions", "periods": periods, **overrides}}

    def by_name(self, source):
        return {metric["name"]: metric for metric in calculate_metrics([source], "finance")}

    def test_decimal_financial_arithmetic_and_provenance(self):
        source = self.source([
            {"period": "FY2024", "revenue": "0.1", "net_income": "0.02", "cash": 3, "debt": 2, "operating_cash_flow": "0.3", "capex": "0.1"},
            {"period": "FY2025", "revenue": "0.3", "net_income": "0.06", "cash": 4, "debt": 2, "operating_cash_flow": "0.4", "capex": "0.1"},
        ])
        metrics = self.by_name(source)
        self.assertEqual(metrics["FY2024: Free cash flow"]["value"], 0.2)
        self.assertEqual(metrics["FY2025: Net profit margin"]["value"], 20)
        self.assertEqual(metrics["FY2025: Net debt"]["value"], -2)
        self.assertEqual(metrics["FY2025: Debt-to-cash ratio"]["value"], 0.5)
        self.assertEqual(metrics["FY2025: Revenue change vs previous listed period"]["value"], 200)
        self.assertEqual(metrics["FY2025: Revenue"]["unit"], "USD millions")
        self.assertTrue(all(item["source_id"] == "financials" and item["formula"] for item in metrics.values()))
        json.dumps(list(metrics.values()), allow_nan=False)

    def test_zero_denominators_stay_explicitly_unavailable(self):
        source = self.source([{"period": "Prior", "revenue": 0, "net_income": 1, "cash": 0, "debt": 5}, {"period": "Current", "revenue": 5}])
        metrics = self.by_name(source)
        for name in ("Prior: Net profit margin", "Prior: Debt-to-cash ratio", "Current: Revenue change vs previous listed period"):
            self.assertIsNone(metrics[name]["value"])
            self.assertIn("denominator is zero", metrics[name]["formula"])

    def test_missing_capex_does_not_assume_zero_spending(self):
        metrics = self.by_name(self.source([{"period": "FY2025", "operating_cash_flow": 10}]))
        self.assertIsNone(metrics["FY2025: Free cash flow"]["value"])
        self.assertIn("missing", metrics["FY2025: Free cash flow"]["formula"])

    def test_units_are_never_guessed(self):
        metrics = self.by_name(self.source([{"period": "FY2025", "revenue": 100, "net_income": 20}], currency=None))
        self.assertTrue(all(item["value"] is None for item in metrics.values()))
        self.assertEqual(metrics["FY2025: Revenue"]["unit"], "unspecified")
        self.assertIn("currency or scale", metrics["FY2025: Net profit margin"]["formula"])

    def test_malformed_units_and_extreme_numeric_exponents_are_missing(self):
        metrics = self.by_name(self.source([{"period": "FY2025", "revenue": "1e-1000000", "net_income": 20}], scale=[]))
        self.assertTrue(all(item["value"] is None for item in metrics.values()))

    def test_nonfinite_boolean_and_negative_capex_are_not_numeric_evidence(self):
        metrics = self.by_name(self.source([{"period": "FY2025", "revenue": True, "net_income": "NaN", "cash": float("inf"), "operating_cash_flow": 10, "capex": -4}]))
        for label in ("Revenue", "Net income", "Cash", "Capital expenditure", "Free cash flow"):
            self.assertIsNone(metrics[f"FY2025: {label}"]["value"])
        json.dumps(list(metrics.values()), allow_nan=False)

    def test_general_and_unstructured_sources_have_no_financial_metrics(self):
        self.assertEqual(calculate_metrics([self.source([{"period": "FY2025", "revenue": 100}])], "general"), [])
        self.assertEqual(calculate_metrics([{"id": "prose", "text": "Revenue increased."}], "finance"), [])
        self.assertEqual(calculate_metrics([self.source([None, {"revenue": 100}])], "finance"), [])

    def test_domain_prompts_require_citations_and_synthetic_labels(self):
        for domain in ("finance", "general"):
            guidance = domain_instructions(domain)
            self.assertIn("source IDs", guidance)
            self.assertIn("synthetic", guidance)
            self.assertIn("untrusted", guidance)
        with self.assertRaises(ValueError):
            domain_instructions("unknown")


if __name__ == "__main__":
    unittest.main()
