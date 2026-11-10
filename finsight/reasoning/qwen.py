"""Qwen served by vLLM (OpenAI-compatible HTTP), called with plain httpx (FinSight provider).

The same chat template settings, decoding parameters and JSON schema are used
for the unchanged and the fine-tuned model; only ``model`` differs (the base
checkpoint name or the LoRA adapter name registered with ``--lora-modules``).
Every sampling parameter is sent explicitly, so the model's generation_config.json
defaults never apply. Reach the server through an SSH tunnel:
``ssh -N -L 8000:127.0.0.1:8000 gpu-wsl``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx

from finsight.contracts import ModelAttempt
from finsight.reasoning.base import Generation


class QwenProvider:
    provider = "qwen"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000", *, label: str | None = None,
                 max_tokens: int = 1024, enable_thinking: bool = False, timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None, family: str = "qwen",
                 chat_template_kwargs: dict | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.config_id = label or f"qwen-{model.replace('/', '_')}"
        self.sampling = {"temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0, "presence_penalty": 0,
                         "frequency_penalty": 0, "repetition_penalty": 1, "seed": 0, "n": 1}
        # Qwen's template takes enable_thinking; other families get only the template kwargs given explicitly (for
        # Gemma 4, thinking stays off when its thinking trigger is not requested; verified from the rendered prompt).
        self.family = family
        self.template_kwargs = {"enable_thinking": enable_thinking} if family == "qwen" else dict(chat_template_kwargs or {})
        self.settings = {"provider": "qwen", "runtime": "vllm", "model": model, "max_tokens": str(max_tokens),
                         "enable_thinking": str(enable_thinking).lower(), "decoding": "greedy, guided JSON schema",
                         "sampling": json.dumps(self.sampling, sort_keys=True)}
        if family != "qwen":
            self.settings.update(provider="vllm", family=family, enable_thinking="not requested",
                                 chat_template_kwargs=json.dumps(self.template_kwargs, sort_keys=True))
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def server_info(self) -> dict:
        """vLLM version and the served models (recorded in the run config; a change blocks resuming)."""
        version = self.client.get("/version")
        version.raise_for_status()
        models = self.client.get("/v1/models")
        models.raise_for_status()
        served = [{k: m.get(k) for k in ("id", "root", "parent", "max_model_len")} for m in models.json().get("data", [])]
        return {"vllm_version": version.json().get("version"), "models": served}

    def server_models(self) -> list[str]:
        response = self.client.get("/v1/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    def generate(self, system: str, user: str, schema: dict, *, job_id=None, stage: str = "answer",
                 attempt: int = 1, scope: str = "finsight") -> Generation:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.sampling,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
        }
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
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
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                        error=f"HTTP {response.status_code}: {response.text[:300]}")])
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
        return Generation(text="".join(parts) if status == "ok" else None, attempts=[attempt], raw="".join(parts))
