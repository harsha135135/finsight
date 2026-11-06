"""Validate user-authored jobs before any model or network activity."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Job:
    goal: str
    sources: list[dict]
    domain: str = "finance"
    provider: str = "demo"
    model: str | None = None
    allowed_hosts: list[str] = field(default_factory=list)
    max_model_calls: int = 12
    max_retries: int = 2
    max_revisions: int = 1
    min_sources: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        if not isinstance(data, dict):
            raise ValueError("Job must be a JSON object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown job fields: {', '.join(sorted(unknown))}")
        try:
            job = cls(**data)
        except TypeError as exc:
            raise ValueError("Job requires goal and sources") from exc
        if not isinstance(job.goal, str) or not 1 <= len(job.goal.strip()) <= 4000:
            raise ValueError("goal must contain 1–4000 characters")
        if job.domain not in ("finance", "general"):
            raise ValueError("domain must be finance or general")
        if job.provider not in ("demo", "openai"):
            raise ValueError("provider must be demo or openai")
        if job.model is not None and (not isinstance(job.model, str) or not job.model.strip()):
            raise ValueError("model must be a nonempty string")
        for key, lower, upper in (
            ("max_model_calls", 1, 100), ("max_retries", 0, 5),
            ("max_revisions", 0, 3), ("min_sources", 1, 10),
        ):
            value = getattr(job, key)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{key} must be an integer between {lower} and {upper}")
        if not isinstance(job.sources, list) or not 1 <= len(job.sources) <= 10:
            raise ValueError("sources must contain 1–10 source definitions")
        if job.min_sources > len(job.sources):
            raise ValueError("min_sources cannot exceed the number of configured sources")
        ids = set()
        for source in job.sources:
            if not isinstance(source, dict) or set(source) - {"id", "title", "path", "url"}:
                raise ValueError("Each source accepts only id, title, and path or url")
            sid = source.get("id")
            if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid):
                raise ValueError("Source id must be 1–64 letters, digits, underscores or hyphens")
            if sid.lower() in ids:
                raise ValueError(f"Duplicate source id: {sid}")
            ids.add(sid.lower())
            if ("path" in source) == ("url" in source):
                raise ValueError(f"Source {sid} requires exactly one of path or url")
            if any(not isinstance(v, str) or not v.strip() for v in source.values()):
                raise ValueError(f"Source {sid} fields must be nonempty strings")
        if not isinstance(job.allowed_hosts, list) or any(
            not isinstance(host, str) or not re.fullmatch(r"[a-z0-9.-]+", host)
            or host.startswith(".") or host.endswith(".") for host in job.allowed_hosts
        ):
            raise ValueError("allowed_hosts must list exact lowercase hostnames without schemes or paths")
        return job


def load_job(path: str | Path) -> tuple[Job, Path]:
    path = Path(path).resolve()
    if path.stat().st_size > 100_000:
        raise ValueError("Job file is too large")
    return Job.from_dict(json.loads(path.read_text(encoding="utf-8"))), path.parent
