"""Durability, run locking, and transactional schedule claims."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from aiagents.store import BudgetExceeded, RunBusy, Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.state = self.base / "state"
        self.store = Store(self.state)
        self.addCleanup(self.store.close)
        self.job = {"goal": "Research supplied evidence", "sources": [{"id": "company", "path": "source.txt"}]}

    def schedule(self):
        with patch("aiagents.store.time.time", return_value=1000):
            return self.store.add_schedule("daily research", self.job, self.base, 60)

    def test_job_snapshot_is_independent_of_callers_mutation(self):
        run_id = self.store.create(self.job, self.base)
        self.job["sources"][0]["path"] = "different.txt"
        self.assertEqual(self.store.get(run_id)["job"]["sources"][0]["path"], "source.txt")

    def test_call_count_and_attempt_event_commit_together(self):
        run_id = self.store.create(self.job, self.base)
        with patch.object(self.store, "_event", side_effect=OSError("simulated event write failure")):
            with self.assertRaises(OSError):
                self.store.consume_call(run_id, "plan", 1)
        self.assertEqual(self.store.get(run_id)["calls"], 0)
        self.assertEqual(self.store.events(run_id), [])
        self.store.consume_call(run_id, "plan", 1)
        other = Store(self.state)
        self.addCleanup(other.close)
        with self.assertRaises(BudgetExceeded):
            other.consume_call(run_id, "plan", 1)
        self.assertEqual(other.get(run_id)["calls"], 1)
        self.assertEqual([e["details"] for e in other.events(run_id)], [{"stage": "plan"}])

    def test_run_lock_is_exclusive_and_released_on_exception(self):
        run_id = self.store.create(self.job, self.base)
        other = Store(self.state)
        self.addCleanup(other.close)
        with self.assertRaisesRegex(RuntimeError, "interrupted task"):
            with self.store.lock(run_id):
                with self.assertRaises(RunBusy):
                    with other.lock(run_id):
                        self.fail("A duplicate executor acquired the lock")
                raise RuntimeError("interrupted task")
        with other.lock(run_id):
            self.assertEqual(other.get(run_id)["status"], "queued")

    def test_invalid_run_ids_cannot_create_lock_paths(self):
        for run_id in ("../outside", "", "a" * 31, "G" * 32):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    with self.store.lock(run_id):
                        self.fail("An invalid run ID acquired a lock")

    def test_schedule_claim_and_next_due_rollback_together(self):
        schedule_id = self.schedule()
        insert = self.store._insert_run

        def interrupted_insert(*args, **kwargs):
            insert(*args, **kwargs)
            raise OSError("simulated crash before advancing schedule")

        with patch.object(self.store, "_insert_run", side_effect=interrupted_insert):
            with self.assertRaises(OSError):
                self.store.enqueue_due(now=1000)
        self.assertEqual(self.store.runs(), [])
        self.assertEqual(self.store.schedules()[0]["next_due"], 1000)
        runs = self.store.enqueue_due(now=1000)
        self.assertEqual(len(runs), 1)
        self.assertEqual(self.store.get(runs[0])["schedule_id"], schedule_id)
        self.assertEqual(self.store.get(runs[0])["due_time"], 1000)
        self.assertEqual(self.store.schedules()[0]["next_due"], 1060)

    def test_concurrent_workers_enqueue_only_one_run(self):
        self.schedule()
        ready = threading.Barrier(2)

        def enqueue():
            store = Store(self.state)
            try:
                ready.wait(timeout=5)
                return store.enqueue_due(now=1000)
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(lambda _: enqueue(), range(2)))
        self.assertEqual(sum(len(result) for result in results), 1)
        self.assertEqual(len(self.store.runs()), 1)
        self.assertEqual(self.store.schedules()[0]["next_due"], 1060)

    def test_unfinished_schedule_run_prevents_overlapping_work(self):
        self.schedule()
        [run_id] = self.store.enqueue_due(now=1000)
        for now, status in ((1060, "queued"), (1120, "running"), (1180, "interrupted")):
            with self.subTest(status=status):
                self.store.set_status(run_id, status)
                self.assertEqual(self.store.enqueue_due(now=now), [])
                self.assertEqual(self.store.pending_scheduled(), [run_id])
                self.assertEqual(self.store.schedules()[0]["next_due"], now + 60)
        self.store.set_status(run_id, "completed")
        next_runs = self.store.enqueue_due(now=1240)
        self.assertEqual(len(next_runs), 1)
        self.assertNotEqual(next_runs[0], run_id)

    def test_missed_intervals_are_coalesced_into_one_run(self):
        self.schedule()
        [run_id] = self.store.enqueue_due(now=10000)
        self.assertEqual(self.store.get(run_id)["due_time"], 1000)
        self.assertEqual(self.store.schedules()[0]["next_due"], 10060)
        self.assertEqual(self.store.enqueue_due(now=10000), [])
        self.assertEqual(len(self.store.runs()), 1)

    def test_disabling_schedule_preserves_queued_work_but_creates_no_new_run(self):
        self.schedule()
        [run_id] = self.store.enqueue_due(now=1000)
        self.store.disable_schedule("daily research")
        self.assertEqual(self.store.pending_scheduled(), [run_id])
        self.store.set_status(run_id, "completed")
        self.assertEqual(self.store.enqueue_due(now=2000), [])
        self.assertEqual(len(self.store.runs()), 1)

    def test_schedule_occurrence_has_database_uniqueness_constraint(self):
        schedule_id = self.schedule()
        [run_id] = self.store.enqueue_due(now=1000)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.db:
                self.store._insert_run(self.job, self.base, schedule_id, 1000)
        self.assertEqual(self.store.pending_scheduled(), [run_id])


if __name__ == "__main__":
    unittest.main()
