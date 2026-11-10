# Scorecard: qwen3-4b-base on finqa-grouped-v2/val_dev_eval

- Manifest sha256: `cbe78f1c576da016` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `aca1f5a3b525`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.1378 | 0.0935 – 0.1878 | 254 |
| abstention_recall_unanswerable | 0.4000 | 0.2424 – 0.5500 | 40 |
| false_abstention_answerable | 0.0276 | 0.0077 – 0.0533 | 254 |
| invalid_output_rate | 0.2007 | 0.1419 – 0.2607 | 294 |
| evidence_validation_failure_rate | 0.4456 | 0.3883 – 0.5037 | 294 |
| operational_failure_rate | 0.0102 | 0.0000 – 0.0228 | 294 |
| diagnostic_lenient_quote_accuracy_answerable | 0.1850 | 0.1331 – 0.2383 | 254 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 23, "ANSWERED": 78, "MODEL_OUTPUT_INVALID": 59, "OPERATIONAL": 3, "VALIDATION_CONTRADICTION": 131}
Evidence-validation failures by subcategory (items; one item can have several): {"adds or subtracts mixed units": 1, "figure is in a different passage than the one cited": 27, "quote formatting differs from the passage (e.g. added $ or commas)": 27, "quoted text not found in the evidence": 10, "stated answer disagrees with the executed program": 40, "value or scale disagrees with the quote": 28}
Latency ms: {'p50': 7496, 'p95': 11833} · TTFT ms: {'p50': 231, 'p95': 375} · tokens: {'input': 532943, 'output': 118661} · cost USD: unavailable
Throughput: {'items_per_second': 0.1298, 'output_tokens_per_second': 52.37, 'wall_seconds': 2265.8, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "qwen3-4b-finqa-v1", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v1/adapter"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 3
