# Scorecard: qwen3-4b-finqa-v2b on finqa-grouped-v2/val_dev_eval

- Manifest sha256: `cbe78f1c576da016` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `6fa4023a34ee`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "qwen3-4b-finqa-v2b", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.5315 | 0.4683 – 0.5964 | 254 |
| abstention_recall_unanswerable | 0.4500 | 0.2857 – 0.6250 | 40 |
| false_abstention_answerable | 0.0354 | 0.0127 – 0.0595 | 254 |
| invalid_output_rate | 0.0034 | 0.0000 – 0.0112 | 294 |
| evidence_validation_failure_rate | 0.1395 | 0.0939 – 0.1912 | 294 |
| operational_failure_rate | 0.0000 | 0.0000 – 0.0000 | 294 |
| diagnostic_lenient_quote_accuracy_answerable | 0.5315 | 0.4683 – 0.5964 | 254 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 27, "ANSWERED": 225, "MODEL_OUTPUT_INVALID": 1, "VALIDATION_CONTRADICTION": 41}
Evidence-validation failures by subcategory (items; one item can have several): {"figure is in a different passage than the one cited": 5, "quote formatting differs from the passage (e.g. added $ or commas)": 2, "quote is not a single figure": 1, "quoted text not found in the evidence": 1, "stated answer disagrees with the executed program": 30, "value or scale disagrees with the quote": 4}
Latency ms: {'p50': 3798, 'p95': 4808} · TTFT ms: {'p50': 258, 'p95': 387} · tokens: {'input': 532943, 'output': 49554} · cost USD: unavailable
Throughput: {'items_per_second': 0.2751, 'output_tokens_per_second': 46.37, 'wall_seconds': 1068.6, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "qwen3-4b-finqa-v1", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v1/adapter"}, {"id": "qwen3-4b-finqa-v2a", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v2a/adapter"}, {"id": "qwen3-4b-finqa-v2b", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v2b/adapter"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 0
