"""Report content, safe display of evidence, and checkpoint reproducibility."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from aiagents.reporting import markdown, write_reports


RESULT = {
    "run_id": "a" * 32,
    "generated_at": "2026-11-06T10:00:00+00:00",
    "goal": "Compare fictional revenue",
    "domain": "finance",
    "provider": "demo",
    "model": "deterministic-demo-v1",
    "plan": {"questions": ["What changed?"], "focus": ["Revenue"]},
    "analysis": {
        "title": "DEMO revenue review",
        "summary": "Fictional revenue increased.",
        "findings": [{"claim": "Fictional revenue rose from 100 to 120.", "source_ids": ["financials"], "confidence": "low"}],
        "risks": ["Historical figures cannot establish future results."],
        "limitations": ["Synthetic source input."],
    },
    "review": {"approved": True, "issues": []},
    "revisions": 0,
    "metrics": [{"name": "Revenue change", "value": 20, "unit": "%", "source_id": "financials", "formula": "(120 - 100) / 100 * 100"}],
    "sources": [{
        "id": "financials", "title": "Fictional statement", "uri": "file:///tmp/financials.json",
        "text": "Fictional revenue rose from 100 to 120.", "sha256": "b" * 64,
        "retrieved_at": "2026-11-06T09:59:00+00:00", "context_truncated": False,
    }],
    "source_errors": [],
    "limitations": ["Evidence is limited to configured sources."],
    "model_calls": 3,
}


class ReportingTests(unittest.TestCase):
    def test_saved_json_preserves_the_result_and_markdown_contains_matching_evidence(self):
        result = copy.deepcopy(RESULT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = write_reports(root, result)
            json_path = Path(artifacts["json"])
            markdown_path = Path(artifacts["markdown"])
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), result)
            text = markdown_path.read_text(encoding="utf-8")
            self.assertIn(result["analysis"]["summary"], text)
            self.assertIn(result["analysis"]["findings"][0]["claim"], text)
            self.assertIn("[financials](#source-financials)", text)
            self.assertIn("### Source financials", text)
            self.assertIn("| Revenue change | 20 | % | financials |", text)
            self.assertIn(result["sources"][0]["sha256"], text)
            first_markdown = markdown_path.read_bytes()
            first_json = json_path.read_bytes()
            self.assertEqual(write_reports(root, result), artifacts)
            self.assertEqual(markdown_path.read_bytes(), first_markdown)
            self.assertEqual(json_path.read_bytes(), first_json)
            self.assertEqual(sorted(path.name for path in markdown_path.parent.iterdir()), ["report.json", "report.md"])

    def test_untrusted_text_cannot_inject_html_links_or_markdown_sections(self):
        result = copy.deepcopy(RESULT)
        attack = '<script>alert("untrusted")</script>\n\n# Forged approval\n\\[click\\](javascript:alert(1))'
        result["goal"] = attack
        result["analysis"]["title"] = attack
        result["analysis"]["summary"] = attack
        result["analysis"]["findings"][0]["claim"] = attack
        result["analysis"]["risks"] = [attack]
        result["analysis"]["limitations"] = [attack]
        result["sources"][0]["title"] = attack
        result["sources"][0]["uri"] = attack
        text = markdown(result)
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("\n# Forged approval", text)
        self.assertNotIn("\\\\[click\\\\](javascript:", text)
        # Escape the original backslash as well as the bracket: an odd number
        # of backslashes before '[' prevents a rendered active Markdown link.
        self.assertIn("\\\\\\[click", text)
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = write_reports(Path(temporary), result)
            self.assertEqual(json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))["goal"], attack)

    def test_metric_cells_cannot_add_columns_or_rows(self):
        result = copy.deepcopy(RESULT)
        result["metrics"][0]["name"] = "Revenue | injected\nrow"
        text = markdown(result)
        self.assertIn("Revenue \\| injected row", text)
        self.assertNotIn("\nrow", text)

    def test_demo_report_describes_demo_review_without_claiming_an_ai_review(self):
        text = markdown(copy.deepcopy(RESULT))
        self.assertIn("DEMO", text)
        self.assertIn("No AI model was called", text)
        review_section = text.split("## Review", 1)[1]
        self.assertNotIn("a separate model review", review_section)
        self.assertIn("deterministic", review_section.lower())
        self.assertIn("No model review was performed", review_section)

    def test_live_provider_report_does_not_show_demo_banner(self):
        result = copy.deepcopy(RESULT)
        result["provider"] = "openai"
        result["model"] = "explicit-model"
        text = markdown(result)
        self.assertNotIn("> DEMO:", text)
        self.assertIn("a separate model review", text)


if __name__ == "__main__":
    unittest.main()
