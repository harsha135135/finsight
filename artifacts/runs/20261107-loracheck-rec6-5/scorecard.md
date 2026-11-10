# Scorecard: rec6-adapter-check on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `3f9ef0a8e366`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "rec6", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.0000 | 0.0000 – 0.0000 | 3 |
| abstention_recall_unanswerable | 0.0000 | 0.0000 – 0.0000 | 2 |
| false_abstention_answerable | 0.0000 | 0.0000 – 0.0000 | 3 |
| invalid_output_rate | 0.0000 | 0.0000 – 0.0000 | 5 |
| evidence_validation_failure_rate | 0.6000 | 0.3333 – 1.0000 | 5 |
| operational_failure_rate | 0.4000 | 0.0000 – 0.6667 | 5 |
| diagnostic_lenient_quote_accuracy_answerable | 0.0000 | 0.0000 – 0.0000 | 3 |

Reason codes: {"OPERATIONAL": 2, "VALIDATION_CONTRADICTION": 3}
Evidence-validation failures by subcategory (items; one item can have several): {"quote is not a single figure": 1, "stated answer disagrees with the executed program": 2}
Latency ms: {'p50': 6596, 'p95': 20802} · TTFT ms: {'p50': 191, 'p95': 351} · tokens: {'input': 7556, 'output': 2920} · cost USD: unavailable
Throughput: {'items_per_second': 0.084, 'output_tokens_per_second': 49.05, 'wall_seconds': 59.5, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "rec6", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/recovery/20261107/uninterrupted/adapter"}, {"id": "zero", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/zero-check"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 2
