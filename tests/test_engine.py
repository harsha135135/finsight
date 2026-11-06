"""Orchestration invariants, exercised without a network or a paid model."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiagents.config import Job
from aiagents.engine import Engine
from aiagents.providers import DemoProvider, RetryableProviderError
from aiagents.reporting import _atomic_write
from aiagents.store import RunBusy, Store


class ScriptedProvider:
    model = "test-model"

    def __init__(self, **actions):
        self.actions = {stage: list(items) for stage, items in actions.items()}
        self.calls = []

    def generate(self, stage, payload, schema):
        self.calls.append((stage, deepcopy(payload)))
        actions = self.actions.get(stage, [])
        action = actions.pop(0) if actions else None
        if isinstance(action, BaseException):
            raise action
        if action is not None:
            return deepcopy(action)
        return DemoProvider().generate(stage, payload, schema)


class EngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source = self.base / "evidence.txt"
        self.source.write_text("Synthetic company revenue increased from 100 to 120.", encoding="utf-8")
        self.state = self.base / "state"
        self.store = Store(self.state)
        self.addCleanup(lambda: self.store.close())

    def create(self, **overrides):
        data = {
            "goal": "Assess supplied evidence",
            "sources": [{"id": "company", "path": "evidence.txt", "title": "Synthetic company evidence"}],
            **overrides,
        }
        return self.store.create(Job.from_dict(data).to_dict(), self.base)

    def execute(self, run_id, provider):
        return Engine(self.store, provider=provider, sleep=lambda _: None).execute(run_id)

    def reopen(self):
        self.store.close()
        self.store = Store(self.state)

    def test_success_persists_final_and_both_reports(self):
        run_id = self.create()
        provider = ScriptedProvider()
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([stage for stage, _ in provider.calls], ["plan", "analyze", "review"])
        final = self.store.checkpoint(run_id, "final")
        self.assertEqual(final["model_calls"], 3)
        self.assertEqual(final["sources"][0]["id"], "company")
        self.assertEqual(json.loads(Path(result["artifacts"]["json"]).read_text()), final)
        self.assertIn("#source-company", Path(result["artifacts"]["markdown"]).read_text())

    def test_total_budget_stops_before_an_unbudgeted_call(self):
        run_id = self.create(max_model_calls=2)
        provider = ScriptedProvider()
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual([stage for stage, _ in provider.calls], ["plan", "analyze"])
        self.assertEqual(self.store.get(run_id)["calls"], 2)
        self.assertIsNone(self.store.checkpoint(run_id, "final"))

    def test_stage_attempt_budget_survives_interruption_and_reopen(self):
        run_id = self.create(max_retries=1)
        provider = ScriptedProvider(plan=[RetryableProviderError("temporary failure"), KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.execute(run_id, provider)
        self.assertEqual(self.store.get(run_id)["status"], "interrupted")
        self.assertEqual(self.store.get(run_id)["calls"], 2)
        self.reopen()
        replacement = ScriptedProvider()
        result = self.execute(run_id, replacement)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("attempt limit", result["error"])
        self.assertEqual(replacement.calls, [])
        self.assertEqual(self.store.get(run_id)["calls"], 2)

    def test_resume_reuses_checkpoints_and_original_source_snapshot(self):
        run_id = self.create(max_retries=1)
        with self.assertRaises(KeyboardInterrupt):
            self.execute(run_id, ScriptedProvider(analyze=[KeyboardInterrupt()]))
        original = self.store.checkpoint(run_id, "collection")
        self.source.write_text("Changed evidence should belong to a new run.", encoding="utf-8")
        self.reopen()
        provider = ScriptedProvider()
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([stage for stage, _ in provider.calls], ["analyze", "review"])
        self.assertEqual(provider.calls[0][1]["sources"][0]["text"], original["sources"][0]["text"])
        self.assertEqual(self.store.get(run_id)["calls"], 4)

    def test_retryable_failure_then_success_counts_both_requests(self):
        run_id = self.create(max_retries=1)
        provider = ScriptedProvider(plan=[RetryableProviderError("temporary failure")])
        self.assertEqual(self.execute(run_id, provider)["status"], "completed")
        self.assertEqual([stage for stage, _ in provider.calls], ["plan", "plan", "analyze", "review"])
        self.assertEqual(self.store.get(run_id)["calls"], 4)
        self.assertIn("temporary failure", provider.calls[1][1]["validation_feedback"])

    def test_unknown_citation_blocks_before_review_or_report(self):
        analysis = DemoProvider().generate("analyze", {"sources": [{"id": "invented", "text": "unsupported"}]}, {})
        run_id = self.create(max_retries=0)
        provider = ScriptedProvider(analyze=[analysis])
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("cite available", result["error"])
        self.assertEqual([stage for stage, _ in provider.calls], ["plan", "analyze"])
        self.assertFalse((self.state / "reports" / run_id).exists())

    def test_unresolved_review_exhausts_only_configured_revisions(self):
        rejected = {"approved": False, "issues": ["The claim overstates the supplied evidence."]}
        run_id = self.create(max_revisions=1)
        provider = ScriptedProvider(review=[rejected, rejected])
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual([stage for stage, _ in provider.calls], ["plan", "analyze", "review", "analyze", "review"])
        revision_payload = provider.calls[3][1]
        self.assertEqual(revision_payload["review_issues"], rejected["issues"])
        self.assertIn("previous_analysis", revision_payload)
        self.assertIsNone(self.store.checkpoint(run_id, "final"))
        self.assertFalse((self.state / "reports" / run_id).exists())
        before = self.store.get(run_id)["calls"]
        self.assertEqual(self.execute(run_id, ScriptedProvider())["status"], "blocked")
        self.assertEqual(self.store.get(run_id)["calls"], before)

    def test_inconsistent_approved_review_cannot_publish(self):
        run_id = self.create(max_retries=0)
        provider = ScriptedProvider(review=[{"approved": True, "issues": ["Unsupported financial claim"]}])
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(self.store.checkpoint(run_id, "final"))

    def test_missing_or_empty_evidence_blocks_and_records_source_errors(self):
        self.source.write_text(" \n\t", encoding="utf-8")
        run_id = self.create(sources=[{"id": "empty", "path": "evidence.txt"}, {"id": "absent", "path": "missing.txt"}])
        provider = ScriptedProvider()
        result = self.execute(run_id, provider)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual([stage for stage, _ in provider.calls], ["plan"])
        collection = self.store.checkpoint(run_id, "collection")
        self.assertEqual(collection["sources"], [])
        self.assertEqual({error["id"] for error in collection["errors"]}, {"empty", "absent"})
        source_events = [event["details"] for event in self.store.events(run_id) if event["kind"] == "source_error"]
        self.assertEqual(source_events, collection["errors"])

    def test_interrupted_report_pair_can_be_regenerated_without_model_calls(self):
        run_id = self.create()
        writes = []

        def fail_second_write(path, content):
            writes.append(path)
            if len(writes) == 2:
                raise OSError("simulated full disk")
            return _atomic_write(path, content)

        with patch("aiagents.reporting._atomic_write", side_effect=fail_second_write):
            result = self.execute(run_id, ScriptedProvider())
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(self.store.checkpoint(run_id, "final"))
        self.assertTrue((self.state / "reports" / run_id / "report.md").exists())
        self.assertFalse((self.state / "reports" / run_id / "report.json").exists())
        self.reopen()
        provider = ScriptedProvider()
        recovered = self.execute(run_id, provider)
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.store.get(run_id)["calls"], 3)
        self.assertEqual(json.loads(Path(recovered["artifacts"]["json"]).read_text()), self.store.checkpoint(run_id, "final"))

    def test_failed_regeneration_cannot_leave_completed_status(self):
        run_id = self.create()
        result = self.execute(run_id, ScriptedProvider())
        Path(result["artifacts"]["json"]).unlink()
        with patch("aiagents.engine.write_reports", side_effect=OSError("simulated full disk")):
            failed = self.execute(run_id, ScriptedProvider())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(self.store.get(run_id)["status"], "failed")
        self.assertIn("simulated full disk", failed["error"])

    def test_duplicate_execution_is_rejected_before_provider_call(self):
        run_id = self.create()
        other = Store(self.state)
        self.addCleanup(other.close)
        provider = ScriptedProvider()
        with self.store.lock(run_id):
            with self.assertRaises(RunBusy):
                Engine(other, provider=provider, sleep=lambda _: None).execute(run_id)
        self.assertEqual(provider.calls, [])
        self.assertEqual(other.get(run_id)["calls"], 0)
        self.assertEqual(other.get(run_id)["status"], "queued")
        self.assertEqual(Engine(other, provider=provider, sleep=lambda _: None).execute(run_id)["status"], "completed")

    def test_manual_resume_cannot_overlap_newer_run_of_same_schedule(self):
        job = Job(goal="Assess supplied evidence", sources=[{"id": "company", "path": "evidence.txt"}])
        with patch("aiagents.store.time.time", return_value=1000):
            self.store.add_schedule("research", job.to_dict(), self.base, 60)
        [old_run] = self.store.enqueue_due(now=1000)
        self.store.set_status(old_run, "failed", "Temporary configuration failure")
        [new_run] = self.store.enqueue_due(now=1060)
        other = Store(self.state)
        self.addCleanup(other.close)
        resumed_provider = ScriptedProvider()
        resume_engine = Engine(other, provider=resumed_provider, sleep=lambda _: None)
        new_provider = ScriptedProvider()
        generate = new_provider.generate

        def attempt_manual_resume(stage, payload, schema):
            if stage == "plan":
                self.assertEqual(self.store.get(new_run)["status"], "running")
                with self.assertRaisesRegex(RunBusy, "Schedule"):
                    resume_engine.execute(old_run)
                self.assertEqual(other.get(old_run)["status"], "failed")
                self.assertEqual(other.get(old_run)["calls"], 0)
                self.assertEqual(resumed_provider.calls, [])
            return generate(stage, payload, schema)

        with patch.object(new_provider, "generate", side_effect=attempt_manual_resume):
            self.assertEqual(self.execute(new_run, new_provider)["status"], "completed")
        self.assertEqual(resume_engine.execute(old_run)["status"], "completed")
        self.assertEqual([stage for stage, _ in resumed_provider.calls], ["plan", "analyze", "review"])


if __name__ == "__main__":
    unittest.main()
