# Scorecard: demo-abstain on finqa-grouped/val_200

- Manifest sha256: `04276a76ead604c4` · dataset revision `0f16e2867bef`
- Prompt `answer-v3` sha256 `09a827edd77dfb51` · commit `417cd3ead0fe`
- Settings: {"model": "always-abstain", "provider": "demo"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.0000 | 0.0000 – 0.0000 | 200 |
| false_abstention_answerable | 1.0000 | 1.0000 – 1.0000 | 200 |
| invalid_output_rate | 0.0000 | 0.0000 – 0.0000 | 200 |
| evidence_validation_failure_rate | 0.0000 | 0.0000 – 0.0000 | 200 |
| operational_failure_rate | 0.0000 | 0.0000 – 0.0000 | 200 |
| diagnostic_lenient_quote_accuracy_answerable | 0.0000 | 0.0000 – 0.0000 | 200 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 200}
Evidence-validation failures by subcategory (items; one item can have several): {}
Latency ms: {'p50': 0, 'p95': 0} · TTFT ms: None · tokens: {'input': 0, 'output': 0} · cost USD: 0
Throughput: None · server: null
Missing predictions: 0 · truncated outputs: 0
