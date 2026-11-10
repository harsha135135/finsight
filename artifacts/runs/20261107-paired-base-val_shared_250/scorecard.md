# Scorecard: qwen3-4b-base on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `f0c674c683ff`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.1250 | 0.0810 – 0.1791 | 200 |
| abstention_recall_unanswerable | 0.2600 | 0.1489 – 0.3846 | 50 |
| false_abstention_answerable | 0.0200 | 0.0000 – 0.0457 | 200 |
| invalid_output_rate | 0.1480 | 0.1004 – 0.2008 | 250 |
| evidence_validation_failure_rate | 0.5080 | 0.4526 – 0.5660 | 250 |
| operational_failure_rate | 0.0200 | 0.0041 – 0.0383 | 250 |
| diagnostic_lenient_quote_accuracy_answerable | 0.1500 | 0.1033 – 0.2069 | 200 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 17, "ANSWERED": 64, "MODEL_OUTPUT_INVALID": 37, "OPERATIONAL": 5, "VALIDATION_CONTRADICTION": 127}
Evidence-validation failures by subcategory (items; one item can have several): {"adds or subtracts mixed units": 1, "figure is in a different passage than the one cited": 32, "quote formatting differs from the passage (e.g. added $ or commas)": 22, "quote is not a single figure": 2, "quoted text not found in the evidence": 12, "stated answer disagrees with the executed program": 36, "value or scale disagrees with the quote": 29}
Latency ms: {'p50': 7334, 'p95': 12256} · TTFT ms: {'p50': 231, 'p95': 349} · tokens: {'input': 447599, 'output': 101872} · cost USD: unavailable
Throughput: {'items_per_second': 0.1285, 'output_tokens_per_second': 52.37, 'wall_seconds': 1945.2, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "qwen3-4b-finqa-v1", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v1/adapter"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 5
