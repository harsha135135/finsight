"""Recorded responses for offline tests: replays stored model text by request hash.

The key covers the model, system prompt, user message and schema, so a recording is only
replayed for exactly the request that produced it. A missing recording is an error attempt,
never a silent default.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from finsight.contracts import ModelAttempt
from finsight.reasoning.base import Generation


def request_key(model: str, system: str, user: str, schema: dict) -> str:
    return hashlib.sha256(json.dumps([model, system, user, schema], sort_keys=True).encode()).hexdigest()


class RecordedProvider:
    provider = "recorded"

    def __init__(self, recordings: Path | dict, model: str = "recorded") -> None:
        from finsight.replay import load_recordings

        self.model = model
        raw = load_recordings(Path(recordings)) if not isinstance(recordings, dict) else recordings
        self.recordings = {k: (v if isinstance(v, dict) else {"text": v, "provenance": None}) for k, v in raw.items()}
        self.settings = {"provider": "recorded", "model": model, "recordings": len(self.recordings)}
        self.replayed: list[dict] = []        # provenance of every recording served (for the report and tests)

    def generate(self, system: str, user: str, schema: dict, *, job_id=None, stage: str = "answer",
                 attempt: int = 1, scope: str = "finsight") -> Generation:
        key = request_key(self.model, system, user, schema)
        started = datetime.now(timezone.utc)
        entry = self.recordings.get(key)
        text = entry["text"] if entry else None
        if entry:
            self.replayed.append({"request_sha256": key, "provenance": entry.get("provenance")})
        status, error = (("ok", None) if text is not None else
                         ("error", f"no recording for request {key[:12]} (replay fails closed; no live model is called)"))
        return Generation(text=text, attempts=[ModelAttempt(requested_model=self.model, served_model=self.model,
                                                            stage=stage, attempt=attempt, status=status,
                                                            latency_ms=0, started_at=started, error=error)])

    @staticmethod
    def record(path: Path, model: str, system: str, user: str, schema: dict, text: str) -> None:
        data = json.loads(path.read_text()) if path.exists() else {}
        data[request_key(model, system, user, schema)] = text
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
