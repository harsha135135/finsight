# Scorecard: qwen3-4b-base on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `86c62eb45125`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.0811 | 0.0000 – 0.2051 | 37 |
| abstention_recall_unanswerable | 0.2308 | 0.0000 – 0.5385 | 13 |
| false_abstention_answerable | 0.0000 | 0.0000 – 0.0000 | 37 |
| invalid_output_rate | 0.1400 | 0.0222 – 0.2609 | 50 |
| evidence_validation_failure_rate | 0.5000 | 0.3800 – 0.6200 | 50 |
| operational_failure_rate | 0.0600 | 0.0000 – 0.1379 | 50 |
| diagnostic_lenient_quote_accuracy_answerable | 0.1081 | 0.0000 – 0.2353 | 37 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 3, "ANSWERED": 12, "MODEL_OUTPUT_INVALID": 7, "OPERATIONAL": 3, "VALIDATION_CONTRADICTION": 25}
Evidence-validation failures by subcategory (items; one item can have several): {"figure is in a different passage than the one cited": 9, "quote formatting differs from the passage (e.g. added $ or commas)": 6, "quoted text not found in the evidence": 2, "stated answer disagrees with the executed program": 6, "value or scale disagrees with the quote": 5}
Latency ms: {'p50': 7732, 'p95': 18891} · TTFT ms: {'p50': 216, 'p95': 328} · tokens: {'input': 89065, 'output': 22435} · cost USD: unavailable
Throughput: {'items_per_second': 0.1175, 'output_tokens_per_second': 52.72, 'wall_seconds': 425.6, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 3
