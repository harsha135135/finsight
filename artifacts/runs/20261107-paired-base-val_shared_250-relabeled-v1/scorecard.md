# Scorecard: qwen3-4b-base on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `f0c674c683ff`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.1250 | 0.0810 – 0.1791 | 200 |
| abstention_recall_unanswerable | 0.3929 | 0.2222 – 0.5862 | 28 |
| false_abstention_answerable | 0.0200 | 0.0000 – 0.0457 | 200 |
| invalid_output_rate | 0.1447 | 0.1000 – 0.1937 | 228 |
| evidence_validation_failure_rate | 0.5000 | 0.4402 – 0.5579 | 228 |
| operational_failure_rate | 0.0175 | 0.0042 – 0.0351 | 228 |
| diagnostic_lenient_quote_accuracy_answerable | 0.1500 | 0.1033 – 0.2069 | 200 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 15, "ANSWERED": 62, "MODEL_OUTPUT_INVALID": 33, "OPERATIONAL": 4, "VALIDATION_CONTRADICTION": 114}
Evidence-validation failures by subcategory (items; one item can have several): {"adds or subtracts mixed units": 1, "figure is in a different passage than the one cited": 30, "quote formatting differs from the passage (e.g. added $ or commas)": 20, "quote is not a single figure": 2, "quoted text not found in the evidence": 8, "stated answer disagrees with the executed program": 34, "value or scale disagrees with the quote": 25}
Latency ms: {'p50': 7368, 'p95': 12001} · TTFT ms: {'p50': 237, 'p95': 350} · tokens: {'input': 411058, 'output': 92464} · cost USD: unavailable
Throughput: {'items_per_second': 0.1285, 'output_tokens_per_second': 47.53, 'wall_seconds': 1945.2, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}, {"id": "qwen3-4b-finqa-v1", "max_model_len": null, "parent": "Qwen/Qwen3-4B", "root": "artifacts/adapters/qwen3-4b-finqa-v1/adapter"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 4
