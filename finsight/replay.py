"""Offline replay: re-execute the real pipeline from a committed bundle, with no GPU, network or paid key.

A replay bundle (``evals/demo/replay/``) holds

* ``sec/index.json`` plus gzipped response bodies: every SEC URL the pipeline requested when the bundle was captured
  (the 10-K and the companyfacts snapshot are the original bytes, checked against the job's recorded sha256);
* ``recordings.json``: model responses keyed by the exact request (model, system prompt, user message, schema), each
  with its provenance: the original job and attempt, the model and its reported version, usage and latency, the prompt
  version, and how the request was obtained.

Replay fails closed. An SEC URL that is not in the bundle raises ``ReplayMiss`` (there is no network fallback), and a
request without a recording yields a failed model attempt, so the job fails with an explicit reason; no live model or
API is ever called. Parsing, retrieval, validation, the XBRL cross-check, checkpoints and the report run as usual.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import httpx

REPLAY_USER_AGENT = "FinSight-offline-replay replay@example.invalid"     # never sent anywhere: the transport is local
RECORDINGS_FORMAT = "finsight-recordings-v1"


class ReplayMiss(RuntimeError):
    pass


class ReplayTransport(httpx.BaseTransport):
    """Serves recorded responses by exact URL; anything else raises. ``requests`` lists every URL asked for."""

    def __init__(self, sec_dir: Path) -> None:
        self.dir = Path(sec_dir)
        self.index = json.loads((self.dir / "index.json").read_text())["responses"]
        self.requests: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        entry = self.index.get(url)
        if request.method != "GET" or entry is None:
            raise ReplayMiss(f"no recorded response for {request.method} {url}; replay never falls back to the network")
        body = gzip.decompress((self.dir / entry["file"]).read_bytes())
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ReplayMiss(f"recorded response for {url} does not match its sha256")
        return httpx.Response(200, content=body, headers={"content-type": entry.get("content_type", "application/octet-stream")})


def write_sec_bundle(sec_dir: Path, responses: dict[str, tuple[bytes, str]], notes: dict[str, str] | None = None) -> dict:
    """responses: url -> (body, content_type). Bodies are stored gzipped (mtime 0, so the bytes are reproducible)."""
    sec_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for url, (body, content_type) in sorted(responses.items()):
        digest = hashlib.sha256(body).hexdigest()
        name = f"{digest[:16]}.gz"
        (sec_dir / name).write_bytes(gzip.compress(body, mtime=0))
        index[url] = {"file": name, "sha256": digest, "bytes": len(body), "content_type": content_type,
                      **({"note": notes[url]} if notes and url in notes else {})}
    data = {"format": "finsight-sec-replay-v1", "responses": index}
    (sec_dir / "index.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return data


def load_recordings(path: Path) -> dict[str, dict]:
    """key -> {"text": ..., "provenance": {...}}; also accepts the older plain {key: text} form."""
    data = json.loads(Path(path).read_text())
    if data.get("format") == RECORDINGS_FORMAT:
        return data["entries"]
    return {k: {"text": v, "provenance": None} for k, v in data.items()}


def write_recordings(path: Path, entries: dict[str, dict], note: str) -> None:
    path.write_text(json.dumps({"format": RECORDINGS_FORMAT, "note": note, "entries": entries}, indent=1, sort_keys=True) + "\n")
