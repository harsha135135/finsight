import sys
import time

from finsight.watchdog import run

# A child that heartbeats for about a second and then freezes (stops touching the file but keeps running).
FREEZES = ("import os, pathlib, time; p = pathlib.Path(os.environ['FINSIGHT_HEARTBEAT_FILE'])\n"
           "for _ in range(5):\n    p.touch(); time.sleep(0.2)\ntime.sleep(60)")


def test_frozen_child_is_killed_and_a_new_worker_started(tmp_path):
    t0 = time.monotonic()
    events = run([sys.executable, "-c", FREEZES], tmp_path / "alive", timeout=1.0, poll=0.1, backoff=0.1, max_starts=2)
    kinds = [e["event"] for e in events]
    assert kinds == ["started", "killed", "exited", "started", "killed", "exited"]
    assert all(e["code"] == -9 for e in events if e["event"] == "exited")
    assert time.monotonic() - t0 < 20


def test_lost_authority_exit_restarts_the_worker(tmp_path):
    events = run([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "alive", timeout=5, poll=0.05, backoff=0.05,
                 max_starts=2)
    assert [e.get("code") for e in events if e["event"] == "exited"] == [3, 3]


def test_child_that_never_heartbeats_is_killed_after_the_startup_grace(tmp_path):
    events = run([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "alive", timeout=1, startup_grace=0.5,
                 poll=0.05, backoff=0.05, max_starts=1)
    assert [e["event"] for e in events] == ["started", "killed", "exited"]
