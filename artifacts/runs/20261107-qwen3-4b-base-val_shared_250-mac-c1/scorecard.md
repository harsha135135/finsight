# Scorecard: qwen3-4b-base on finqa-grouped-comparison/val_shared_250

- Manifest sha256: `d1d548d0607048c3` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `86c62eb45125`
- Settings: {"decoding": "greedy, guided JSON schema", "enable_thinking": "false", "max_tokens": "1024", "model": "Qwen/Qwen3-4B", "provider": "qwen", "runtime": "vllm", "sampling": "{\"frequency_penalty\": 0, \"min_p\": 0, \"n\": 1, \"presence_penalty\": 0, \"repetition_penalty\": 1, \"seed\": 0, \"temperature\": 0, \"top_k\": -1, \"top_p\": 1}"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.1300 | 0.0817 – 0.1872 | 200 |
| abstention_recall_unanswerable | 0.3000 | 0.1837 – 0.4314 | 50 |
| false_abstention_answerable | 0.0200 | 0.0000 – 0.0457 | 200 |
| invalid_output_rate | 0.1280 | 0.0850 – 0.1720 | 250 |
| evidence_validation_failure_rate | 0.5240 | 0.4680 – 0.5830 | 250 |
| operational_failure_rate | 0.0360 | 0.0153 – 0.0593 | 250 |
| diagnostic_lenient_quote_accuracy_answerable | 0.1500 | 0.1010 – 0.2092 | 200 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 19, "ANSWERED": 59, "MODEL_OUTPUT_INVALID": 32, "OPERATIONAL": 9, "VALIDATION_CONTRADICTION": 131}
Evidence-validation failures by subcategory (items; one item can have several): {"adds or subtracts mixed units": 1, "figure is in a different passage than the one cited": 33, "quote formatting differs from the passage (e.g. added $ or commas)": 25, "quote is not a single figure": 2, "quoted text not found in the evidence": 11, "stated answer disagrees with the executed program": 40, "value or scale disagrees with the quote": 28}
Latency ms: {'p50': 7381, 'p95': 17082} · TTFT ms: {'p50': 228, 'p95': 364} · tokens: {'input': 447599, 'output': 104280} · cost USD: unavailable
Throughput: {'items_per_second': 0.1258, 'output_tokens_per_second': 52.48, 'wall_seconds': 1987.2, 'sessions': 1, 'concurrency': [1]} · server: {"models": [{"id": "Qwen/Qwen3-4B", "max_model_len": 5120, "parent": null, "root": "Qwen/Qwen3-4B"}], "vllm_version": "0.33.0"}
Missing predictions: 0 · truncated outputs: 9
