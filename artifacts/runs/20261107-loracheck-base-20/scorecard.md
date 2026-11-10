# Scorecard: qwen3-4b-base-loracheck on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `3f9ef0a8e366`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.2000 | 0.0000 – 0.4706 | 15 |
| abstention_recall_unanswerable | 0.2000 | 0.0000 – 0.7500 | 5 |
| false_abstention_answerable | 0.0000 | 0.0000 – 0.0000 | 15 |
| invalid_output_rate | 0.2000 | 0.0000 – 0.5217 | 20 |
| evidence_validation_failure_rate | 0.3000 | 0.1600 – 0.5333 | 20 |
| operational_failure_rate | 0.0500 | 0.0000 – 0.1500 | 20 |
| diagnostic_lenient_quote_accuracy_answerable | 0.2000 | 0.0000 – 0.4706 | 15 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 1, "ANSWERED": 8, "MODEL_OUTPUT_INVALID": 4, "OPERATIONAL": 1, "VALIDATION_CONTRADICTION": 6}
Evidence-validation failures by subcategory (items; one item can have several): {"figure is in a different passage than the one cited": 2, "quote formatting differs from the passage (e.g. added $ or commas)": 1, "value or scale disagrees with the quote": 3}
Latency ms: {'p50': 8495, 'p95': 12321} · TTFT ms: {'p50': 265, 'p95': 383} · tokens: {'input': 36285, 'output': 9413} · cost USD: unavailable
Throughput: {'items_per_second': 0.1114, 'output_tokens_per_second': 52.45, 'wall_seconds': 179.5, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "rec6", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/recovery/20261107/uninterrupted/adapter"}, {"id": "zero", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/zero-check"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 1
