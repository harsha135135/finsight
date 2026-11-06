"""Qwen served by vLLM (OpenAI-compatible HTTP), called with plain httpx.

The same chat template settings, decoding parameters and JSON schema are used
for the unchanged and the fine-tuned model; only ``model`` differs (the base
checkpoint name or the LoRA adapter name registered with ``--lora-modules``).
Reach the server through an SSH tunnel: ``ssh -N -L 8000:localhost:8000 gpu-box``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx

from finsight.contracts import ModelAttempt
from modelforge.adapters import Generation
from modelforge.items import EvalItem


class QwenAdapter:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000", *, label: str | None = None,
                 max_tokens: int = 2048, enable_thinking: bool = False, timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.config_id = label or f"qwen-{model.replace('/', '_')}"
        self.settings = {"provider": "qwen", "runtime": "vllm", "model": model, "temperature": "0",
                         "max_tokens": str(max_tokens), "enable_thinking": str(enable_thinking).lower(),
                         "decoding": "greedy, guided JSON schema"}
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def server_models(self) -> list[str]:
        response = self.client.get("/v1/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        ttft = None
        parts: list[str] = []
        usage: dict = {}
        finish = None
        try:
            with self.client.stream("POST", "/v1/chat/completions", json=body) as response:
                if response.status_code != 200:
                    response.read()
                    status = "retryable_error" if response.status_code in (429, 500, 502, 503, 504) else "error"
                    return Generation(text=None, attempts=[ModelAttempt(
                        requested_model=self.model, stage="answer", attempt=1, status=status, started_at=started,
                        latency_ms=int((time.perf_counter() - t0) * 1000), error=f"HTTP {response.status_code}")])
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            if ttft is None:
                                ttft = int((time.perf_counter() - t0) * 1000)
                            parts.append(delta)
                        finish = choice.get("finish_reason") or finish
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return Generation(text=None, attempts=[ModelAttempt(
                requested_model=self.model, stage="answer", attempt=1, status="uncertain", started_at=started,
                latency_ms=int((time.perf_counter() - t0) * 1000), error=type(exc).__name__)])
        status = "truncated" if finish == "length" else "ok"
        attempt = ModelAttempt(
            requested_model=self.model, served_model=self.model, stage="answer", attempt=1, status=status,
            input_tokens_uncached=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            latency_ms=int((time.perf_counter() - t0) * 1000), ttft_ms=ttft, estimated_cost_usd=None,
            started_at=started,
        )
        return Generation(text="".join(parts) if status == "ok" else None, attempts=[attempt])
