"""Provider contract and transport failures, with no live network or API cost."""

import http.client
import json
import socket
import unittest
from unittest.mock import patch

from aiagents.providers import (
    DemoProvider,
    OpenAIProvider,
    ProviderError,
    RetryableProviderError,
    make_provider,
)


SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}},
    "required": ["approved"],
    "additionalProperties": False,
}
SOURCES = [{"id": "source-1", "title": "Synthetic example", "text": "Fictional revenue was 100."}]


def response_body(value=None, **changes):
    result = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(value or {"approved": True})}],
            }
        ],
    }
    result.update(changes)
    return json.dumps(result).encode()


class OpenAIProviderTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret", "OPENAI_MODEL": "configured-model"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        connection_type = patch("aiagents.providers.http.client.HTTPSConnection")
        self.connection_type = connection_type.start()
        self.addCleanup(connection_type.stop)
        self.connection = self.connection_type.return_value
        self.response = self.connection.getresponse.return_value
        self.response.status = 200
        self.response.read.return_value = response_body()

    def test_request_uses_explicit_model_strict_schema_and_no_storage(self):
        provider = OpenAIProvider("chosen-model", timeout=12)
        output = provider.generate("review", {"sources": SOURCES}, SCHEMA)
        self.assertEqual(output, {"approved": True})
        self.connection_type.assert_called_once_with("api.openai.com", timeout=12)
        self.connection.request.assert_called_once()
        args, kwargs = self.connection.request.call_args
        self.assertEqual(args, ("POST", "/v1/responses"))
        body = json.loads(kwargs["body"])
        self.assertEqual(body["model"], "chosen-model")
        self.assertFalse(body["store"])
        self.assertEqual(body["max_output_tokens"], 4000)
        self.assertEqual(body["text"]["format"], {
            "type": "json_schema", "name": "research_review", "strict": True, "schema": SCHEMA,
        })
        self.assertEqual(json.loads(body["input"]), {"stage": "review", "payload": {"sources": SOURCES}})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-secret")
        self.connection.close.assert_called_once()

    def test_environment_model_used_without_guessing(self):
        self.assertEqual(make_provider("openai").model, "configured-model")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret"}, clear=True):
            with self.assertRaisesRegex(ProviderError, "OPENAI_MODEL"):
                make_provider("openai")
        self.connection_type.assert_not_called()

    def test_missing_key_is_clear_and_does_not_connect(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ProviderError, "OPENAI_API_KEY"):
                OpenAIProvider("configured-model")
        self.connection_type.assert_not_called()

    def test_retryable_http_statuses_make_one_request_and_hide_body(self):
        for status in (429, 500, 502, 503, 599):
            with self.subTest(status=status):
                self.connection.reset_mock()
                self.response.status = status
                self.response.read.return_value = b"private-provider-body test-secret"
                with self.assertRaises(RetryableProviderError) as caught:
                    OpenAIProvider().generate("review", {}, SCHEMA)
                self.assertNotIn("private-provider-body", str(caught.exception))
                self.assertNotIn("test-secret", str(caught.exception))
                self.connection.request.assert_called_once()
                self.response.read.assert_not_called()
                self.connection.close.assert_called_once()

    def test_permanent_errors_and_redirects_are_not_retried(self):
        for status in (301, 307, 400, 401, 403, 404):
            with self.subTest(status=status):
                self.connection.reset_mock()
                self.response.status = status
                with self.assertRaises(ProviderError) as caught:
                    OpenAIProvider().generate("review", {}, SCHEMA)
                self.assertNotIsInstance(caught.exception, RetryableProviderError)
                self.connection.request.assert_called_once()
                self.connection.close.assert_called_once()

    def test_network_failures_are_retryable_and_sanitized(self):
        for failure in (socket.timeout("test-secret"), OSError("test-secret"), http.client.RemoteDisconnected("test-secret")):
            with self.subTest(failure=type(failure).__name__):
                self.connection.reset_mock()
                self.connection.request.side_effect = failure
                with self.assertRaises(RetryableProviderError) as caught:
                    OpenAIProvider().generate("review", {}, SCHEMA)
                self.assertNotIn("test-secret", str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                self.connection.request.assert_called_once()
                self.connection.close.assert_called_once()

    def test_refusal_rejected_even_if_other_text_is_valid(self):
        self.response.read.return_value = response_body(output=[{
            "type": "message", "content": [
                {"type": "output_text", "text": '{"approved":true}'},
                {"type": "refusal", "refusal": "private refusal body"},
            ],
        }])
        with self.assertRaisesRegex(ProviderError, "refused") as caught:
            OpenAIProvider().generate("review", {}, SCHEMA)
        self.assertNotIn("private refusal body", str(caught.exception))

    def test_incomplete_response_never_returns_partial_json(self):
        for changes in (
            {"status": "incomplete"},
            {"status": "failed"},
            {"incomplete_details": {"reason": "max_output_tokens"}},
            {"output": [{"type": "message", "status": "incomplete", "content": []}]},
        ):
            with self.subTest(changes=changes):
                self.response.read.return_value = response_body(**changes)
                with self.assertRaisesRegex(ProviderError, "complete"):
                    OpenAIProvider().generate("review", {}, SCHEMA)

    def test_reasoning_items_are_ignored_and_text_parts_combined(self):
        self.response.read.return_value = response_body(output=[
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [
                {"type": "output_text", "text": '{"approved":'},
                {"type": "output_text", "text": 'true}'},
            ]},
        ])
        self.assertEqual(OpenAIProvider().generate("review", {}, SCHEMA), {"approved": True})

    def test_bad_responses_fail_cleanly(self):
        for raw in (
            b"not json private-body", b"\xff", b"[]",
            response_body(output=[]),
            response_body(output=[{"type": "message", "content": "private-body"}]),
            response_body(output=[{"type": "message", "content": [{"type": "output_text", "text": "[]"}]}]),
            response_body(output=[{"type": "message", "content": [{"type": "output_text", "text": "bad private-body"}]}]),
        ):
            with self.subTest(raw=raw[:60]):
                self.response.read.return_value = raw
                with self.assertRaises(ProviderError) as caught:
                    OpenAIProvider().generate("review", {}, SCHEMA)
                self.assertNotIn("private-body", str(caught.exception))

    def test_oversized_response_rejected(self):
        self.response.read.return_value = b" " * 2_000_001
        with self.assertRaisesRegex(ProviderError, "size limit"):
            OpenAIProvider().generate("review", {}, SCHEMA)
        self.response.read.assert_called_once_with(2_000_001)

    def test_invalid_input_fails_before_network(self):
        with self.assertRaisesRegex(ProviderError, "JSON serializable"):
            OpenAIProvider().generate("review", {"invalid": float("nan")}, SCHEMA)
        with self.assertRaisesRegex(ProviderError, "Unsupported"):
            OpenAIProvider().generate("execute_trade", {}, SCHEMA)
        self.connection_type.assert_not_called()


class DemoProviderTests(unittest.TestCase):
    def test_offline_deterministic_analysis_is_labeled_and_cited(self):
        provider = make_provider("demo")
        with patch("aiagents.providers.http.client.HTTPSConnection") as network:
            payload = {"goal": "Assess fictional revenue", "sources": SOURCES}
            result = provider.generate("analyze", payload, {})
            self.assertEqual(result, provider.generate("analyze", payload, {}))
            self.assertIn("DEMO", result["title"])
            self.assertIn("DEMO", result["summary"])
            self.assertIn("Fictional revenue was 100.", result["findings"][0]["claim"])
            self.assertEqual(result["findings"][0]["source_ids"], ["source-1"])
            self.assertEqual(result["findings"][0]["confidence"], "low")
            self.assertEqual(provider.generate("review", {**payload, "analysis": result}, {}), {"approved": True, "issues": []})
            network.assert_not_called()

    def test_demo_review_rejects_unknown_citations_and_missing_findings(self):
        provider = DemoProvider()
        analysis = provider.generate("analyze", {"sources": SOURCES}, {})
        analysis["findings"][0]["source_ids"] = ["made-up-source"]
        review = provider.generate("review", {"sources": SOURCES, "analysis": analysis}, {})
        self.assertFalse(review["approved"])
        self.assertIn("supporting source IDs", review["issues"][0])
        empty = provider.generate("analyze", {"sources": []}, {})
        self.assertFalse(provider.generate("review", {"analysis": empty}, {})["approved"])

    def test_plan_contract_and_unknown_provider(self):
        plan = DemoProvider().generate("plan", {"goal": "Compare supplied evidence"}, {})
        self.assertEqual(set(plan), {"questions", "focus"})
        self.assertTrue(all(isinstance(question, str) for question in plan["questions"]))
        with self.assertRaisesRegex(ProviderError, "Unknown provider"):
            make_provider("untrusted-endpoint")


if __name__ == "__main__":
    unittest.main()
