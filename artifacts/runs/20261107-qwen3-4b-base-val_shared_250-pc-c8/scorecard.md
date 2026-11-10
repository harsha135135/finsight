# Scorecard: qwen3-4b-base on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `86c62eb45125`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.1250 | 0.0794 – 0.1784 | 200 |
| abstention_recall_unanswerable | 0.3000 | 0.1837 – 0.4314 | 50 |
| false_abstention_answerable | 0.0250 | 0.0048 – 0.0526 | 200 |
| invalid_output_rate | 0.1360 | 0.0898 – 0.1829 | 250 |
| evidence_validation_failure_rate | 0.5280 | 0.4726 – 0.5837 | 250 |
| operational_failure_rate | 0.0240 | 0.0079 – 0.0438 | 250 |
| diagnostic_lenient_quote_accuracy_answerable | 0.1500 | 0.1029 – 0.2050 | 200 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 20, "ANSWERED": 58, "MODEL_OUTPUT_INVALID": 34, "OPERATIONAL": 6, "VALIDATION_CONTRADICTION": 132}
Evidence-validation failures by subcategory (items; one item can have several): {"adds or subtracts mixed units": 1, "figure is in a different passage than the one cited": 32, "quote formatting differs from the passage (e.g. added $ or commas)": 25, "quote is not a single figure": 1, "quoted text not found in the evidence": 11, "stated answer disagrees with the executed program": 40, "value or scale disagrees with the quote": 29}
Latency ms: {'p50': 19623, 'p95': 29443} · TTFT ms: {'p50': 10267, 'p95': 16876} · tokens: {'input': 447599, 'output': 102284} · cost USD: unavailable
Throughput: {'items_per_second': 0.397, 'output_tokens_per_second': 162.43, 'wall_seconds': 629.7, 'sessions': 1, 'concurrency': [8]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 6
