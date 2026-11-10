# Scorecard: qwen3-4b-finqa-v2a on finqa-grouped-v2-wrong-period/val_dev_wrong_period

- Manifest sha256: `3e21a39b02787985` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `6fa4023a34ee`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "qwen3-4b-finqa-v2a", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| abstention_recall_unanswerable | 0.1762 | 0.1105 – 0.2527 | 193 |
| invalid_output_rate | 0.0000 | 0.0000 – 0.0000 | 193 |
| evidence_validation_failure_rate | 0.1088 | 0.0632 – 0.1590 | 193 |
| operational_failure_rate | 0.0000 | 0.0000 – 0.0000 | 193 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 34, "ANSWERED": 138, "VALIDATION_CONTRADICTION": 21}
Evidence-validation failures by subcategory (items; one item can have several): {"figure is in a different passage than the one cited": 1, "quote formatting differs from the passage (e.g. added $ or commas)": 3, "stated answer disagrees with the executed program": 17}
Latency ms: {'p50': 3765, 'p95': 4392} · TTFT ms: {'p50': 274, 'p95': 370} · tokens: {'input': 358369, 'output': 30262} · cost USD: unavailable
Throughput: {'items_per_second': 0.2926, 'output_tokens_per_second': 45.88, 'wall_seconds': 659.6, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "qwen3-4b-finqa-v1", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v1/adapter"}, {"id": "qwen3-4b-finqa-v2a", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v2a/adapter"}, {"id": "qwen3-4b-finqa-v2b", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v2b/adapter"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 0
