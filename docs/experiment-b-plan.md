# Experiment (b): v2a plus wrong-period negatives (written before training, 2026-11-09)

**Question.** Starting from v2a, does adding negatives that ask about a year the evidence does not cover improve
abstention further, and at what cost?

**Data** (`artifacts/training/finqa-grouped-v2b-p10`, train sha256 `66e60794…`). v2a's 1,711 training rows unchanged
(byte-identical), plus 142 wrong-period negatives (10 per 100 answerable examples), each made from a training
question that is in v2a as an answerable example (pairing holds for all 142). Generator `period_shift` version 2:
the question's evidenced year is moved 2-4 years outside every year in the question and evidence (direction seeded
per item), absent as a 4-digit year and as FY/'YY forms; forecasting/hypothetical questions skipped.

| | v2a | v2b |
|---|---|---|
| Answerable | 1,428 | 1,428 (identical rows) |
| Clean (evidence-removed) negatives | 283 | 283 (identical rows) |
| Wrong-period negatives | 0 | 142 |
| Negatives | 283 / 1,711 = 16.5%; 19.8 per 100 answerable | 425 / 1,853 = 22.9%; 29.8 per 100 answerable |

**Fixed:** base revision, QLoRA settings, lr/schedule, seed 42, max length 4,096, **210 optimizer steps** (so v2b sees
its answerable rows about 8% less often: ~1.81 vs ~1.96 epochs; recorded as part of the treatment), checkpoints
every 30 steps, selection by lowest eval loss on the same 237 `val_select` rows (byte-identical file), all
checkpoints kept.

**Manual checks before training** (`evals/labels/train-negatives-sample-check-v1.json`): wrong-period 29/30 verified,
1 ambiguous (after fixing a generator bug the first sample found); clean negatives 28/30 verified, 2 ambiguous.

**Evaluation, fixed now.**
- Primary: `finqa-grouped-v2/val_dev_eval`, default FinQA serve command, one request at a time, paired against v2a,
  v1 and base: correct abstention (40 verified negatives), answerable accuracy, false abstention, plus the 36 answerable
  questions in groups v1's selection never touched, reported separately. Haiku: the completed run is reused; no new
  Claude calls.
- Secondary diagnostic: `finqa-grouped-v2-wrong-period` (196 wrong-period versions of val_dev questions; sample check
  19/20, 1 ambiguous), run for base, v1, v2a and v2b. v2b is trained on this kind of negative and the others are not,
  so this slice shows whether the behaviour was learned, not general abstention; the primary metric stays val_dev_eval.

**Reading.** An accuracy interval that includes zero is "no statistically clear change", not "no cost". With 40
negatives, abstention changes under about 15-20 points may not be distinguishable.
