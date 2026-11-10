# Scorecard: qwen3-4b-finqa-v1 on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `3b44c050e553`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "qwen3-4b-finqa-v1", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.5100 | 0.4389 – 0.5798 | 200 |
| abstention_recall_unanswerable | 0.3200 | 0.2041 – 0.4400 | 50 |
| false_abstention_answerable | 0.0050 | 0.0000 – 0.0153 | 200 |
| invalid_output_rate | 0.0040 | 0.0000 – 0.0128 | 250 |
| evidence_validation_failure_rate | 0.1720 | 0.1308 – 0.2152 | 250 |
| operational_failure_rate | 0.0040 | 0.0000 – 0.0133 | 250 |
| diagnostic_lenient_quote_accuracy_answerable | 0.5100 | 0.4389 – 0.5798 | 200 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 17, "ANSWERED": 188, "MODEL_OUTPUT_INVALID": 1, "OPERATIONAL": 1, "VALIDATION_CONTRADICTION": 43}
Evidence-validation failures by subcategory (items; one item can have several): {"figure is in a different passage than the one cited": 3, "quote formatting differs from the passage (e.g. added $ or commas)": 2, "quote is not a single figure": 1, "quoted text not found in the evidence": 2, "stated answer disagrees with the executed program": 31, "value or scale disagrees with the quote": 5}
Latency ms: {'p50': 3775, 'p95': 4517} · TTFT ms: {'p50': 254, 'p95': 389} · tokens: {'input': 447599, 'output': 43761} · cost USD: unavailable
Throughput: {'items_per_second': 0.2658, 'output_tokens_per_second': 46.52, 'wall_seconds': 940.7, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "qwen3-4b-finqa-v1", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v1/adapter"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 1
