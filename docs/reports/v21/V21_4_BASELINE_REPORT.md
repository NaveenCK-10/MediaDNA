# V21.4 BENCHMARK BASELINE REPORT

## Trivial Baselines
- **All-Real Baseline**: Predicts 0 for everything.
- **All-Fake Baseline**: Predicts 1 for everything.
- **Majority-Class Baseline**: Predicts the most frequent class in the set.

## V21.4 Detector Performance
The detector performance is extracted deterministically from `V21_4_PREDICTIONS_run1.csv`. It outperforms trivial baselines and shows high ROC-AUC on audio anomalies. See `V21_4_BASELINE_RESULTS.json` for precise metric outputs calculated from raw predictions.