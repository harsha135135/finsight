"""Exercise the actual command entry point with isolated state and no network."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CLIWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / "state"
        self.job_dir = self.root / "jobs" / "finance"
        self.job_dir.mkdir(parents=True)
        (self.job_dir / "evidence").mkdir()
        self.source = self.job_dir / "evidence" / "company.txt"
        self.source.write_text("Fictional company: sample revenue increased from 100 to 120 units.", encoding="utf-8")
        self.job = {
            "goal": "Review the fictional company's supplied evidence",
            "domain": "finance",
            "provider": "demo",
            "sources": [{"id": "company", "title": "Fictional company data", "path": "evidence/company.txt"}],
            "max_retries": 0,
        }
        self.job_path = self.job_dir / "job.json"
        self.write_job()

        # Fail immediately if any subprocess accidentally attempts real network
        # access. This also makes missing-key tests independent of connectivity.
        guard = self.root / "network-guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(
            "import socket\n"
            "def no_network(*args, **kwargs):\n"
            "    raise AssertionError('NETWORK_FORBIDDEN_IN_TESTS')\n"
            "socket.getaddrinfo = no_network\n"
            "socket.create_connection = no_network\n"
            "socket.socket.connect = no_network\n",
            encoding="utf-8",
        )
        self.env = os.environ.copy()
        self.env.pop("OPENAI_API_KEY", None)
        self.env.pop("OPENAI_MODEL", None)
        self.env["PYTHONPATH"] = os.pathsep.join([str(guard), str(ROOT)])
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def write_job(self):
        self.job_path.write_text(json.dumps(self.job), encoding="utf-8")

    def cli(self, *arguments, expected=0):
        process = subprocess.run(
            [sys.executable, "-m", "aiagents", "--state-dir", str(self.state), *map(str, arguments)],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(process.returncode, expected, f"stdout: {process.stdout}\nstderr: {process.stderr}")
        self.assertNotIn("Traceback", process.stderr)
        self.assertNotIn("NETWORK_FORBIDDEN_IN_TESTS", process.stdout + process.stderr)
        return process

    def test_run_status_and_resume_regenerate_persisted_report(self):
        execution = json.loads(self.cli("run", self.job_path).stdout)
        self.assertEqual(execution["status"], "completed")
        run_id = execution["run_id"]
        report_path = Path(execution["artifacts"]["json"])
        markdown_path = Path(execution["artifacts"]["markdown"])
        original_json = report_path.read_bytes()
        original_markdown = markdown_path.read_bytes()
        report = json.loads(original_json)
        self.assertEqual(report["sources"][0]["uri"], self.source.as_uri())
        self.assertIn("sample revenue increased", report["sources"][0]["text"])
        self.assertTrue(report["review"]["approved"])
        self.assertEqual(report["review"]["issues"], [])
        self.assertEqual(report["model_calls"], 3)

        status = json.loads(self.cli("status", run_id, "--events").stdout)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["calls"], 3)
        self.assertEqual(Path(status["base_dir"]), self.job_dir)
        self.assertEqual(status["source_errors"], [])
        self.assertEqual(Path(status["report_directory"]), report_path.parent)
        self.assertEqual(sum(event["kind"] == "model_attempt" for event in status["events"]), 3)

        # A completed resume reconstructs saved artifacts without re-reading
        # changed sources or repeating reasoning stages.
        self.source.write_text("Later edited source content must not replace the saved snapshot.", encoding="utf-8")
        report_path.unlink()
        markdown_path.unlink()
        resumed = json.loads(self.cli("resume", run_id).stdout)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(report_path.read_bytes(), original_json)
        self.assertEqual(markdown_path.read_bytes(), original_markdown)
        self.assertEqual(json.loads(self.cli("status", run_id).stdout)["calls"], 3)
        self.assertEqual(len(json.loads(self.cli("runs").stdout)), 1)

    def test_schedule_worker_once_is_persistent_and_does_not_duplicate_runs(self):
        registered = json.loads(self.cli("schedule", "add", "daily-research", self.job_path, "--every", "60").stdout)
        self.assertEqual(registered["status"], "registered")
        schedules = json.loads(self.cli("schedule", "list").stdout)
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0]["name"], "daily-research")
        self.assertEqual(schedules[0]["interval"], 60)
        self.assertEqual(schedules[0]["enabled"], 1)

        execution = json.loads(self.cli("worker", "--once").stdout)
        self.assertEqual(execution["status"], "completed")
        self.assertTrue(Path(execution["artifacts"]["markdown"]).is_file())
        self.assertEqual(self.cli("worker", "--once").stdout, "")
        runs = json.loads(self.cli("runs").stdout)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["schedule_id"], registered["schedule_id"])
        disabled = json.loads(self.cli("schedule", "disable", "daily-research").stdout)
        self.assertEqual(disabled["status"], "disabled")
        self.assertEqual(json.loads(self.cli("schedule", "list").stdout)[0]["enabled"], 0)

    def test_missing_openai_model_explains_configuration_before_creating_run(self):
        self.job["provider"] = "openai"
        self.write_job()
        failed = self.cli("run", self.job_path, expected=2)
        self.assertIn("OPENAI_MODEL", failed.stderr)
        self.assertEqual(json.loads(self.cli("runs").stdout), [])

    def test_missing_openai_key_records_clear_failure_without_model_attempt(self):
        self.job.update(provider="openai", model="explicit-test-model")
        self.write_job()
        result = json.loads(self.cli("run", self.job_path, expected=1).stdout)
        self.assertEqual(result["status"], "failed")
        self.assertIn("OPENAI_API_KEY", result["error"])
        status = json.loads(self.cli("status", result["run_id"], "--events").stdout)
        self.assertEqual(status["calls"], 0)
        self.assertFalse(any(event["kind"] == "model_attempt" for event in status["events"]))
        self.assertNotIn("artifacts", result)


if __name__ == "__main__":
    unittest.main()
