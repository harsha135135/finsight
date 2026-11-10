# Scorecard: claude-claude-haiku-5-5-medium-batch on finqa-grouped-v2/val_dev_eval

- Manifest sha256: `cbe78f1c576da016` · dataset revision `0f16e2867bef`
- Prompt `answer-v4` sha256 `f2d643c5829026e6` · commit `1b03eab9da6e`
- Settings: {"api": "message batches (50% price)", "effort": "medium", "max_tokens": "2600", "model": "claude-haiku-5-5", "provider": "anthropic", "sampling": "none sent", "thinking": "adaptive (model default)"}
- Tolerance: FinQA: executed value rounded to 5 decimals as float equals gold

| Metric | Mean | 95% CI (filing-clustered) | n |
|---|---:|---|---:|
| accuracy_answerable | 0.6024 | 0.5311 – 0.6745 | 254 |
| abstention_recall_unanswerable | 0.9000 | 0.7907 – 0.9773 | 40 |
| false_abstention_answerable | 0.0787 | 0.0415 – 0.1235 | 254 |
| invalid_output_rate | 0.0408 | 0.0180 – 0.0688 | 294 |
| evidence_validation_failure_rate | 0.1020 | 0.0686 – 0.1425 | 294 |
| operational_failure_rate | 0.0000 | 0.0000 – 0.0000 | 294 |
| diagnostic_lenient_quote_accuracy_answerable | 0.6102 | 0.5394 – 0.6800 | 254 |

Reason codes: {"ABSTAIN_EVIDENCE_ABSENT": 56, "ANSWERED": 196, "MODEL_OUTPUT_INVALID": 12, "VALIDATION_CONTRADICTION": 30}
Evidence-validation failures by subcategory (items; one item can have several): {"figure is in a different passage than the one cited": 22, "quote formatting differs from the passage (e.g. added $ or commas)": 2, "stated answer disagrees with the executed program": 1, "value or scale disagrees with the quote": 5}
Latency ms: None · TTFT ms: None · tokens: {'input': 1134396, 'output': 225367} · cost USD: 0.113073
Throughput: {'items_per_second': 927.4448, 'output_tokens_per_second': 710936.91, 'wall_seconds': 0.3, 'sessions': 1, 'concurrency': [1]} · server: null
Missing predictions: 0 · truncated outputs: 0
