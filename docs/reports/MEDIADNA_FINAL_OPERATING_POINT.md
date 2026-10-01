# MediaDNA Final Operating Point

**Date:** 2026-09-28
**Scope:** Final Application Decision Policy vs. Historical Locked-Test Metrics

## 1. Historical Locked-Test Metrics (Threshold: 0.70)
*Status: FROZEN HISTORICAL RECORD*

During the initial scientific evaluation (Phase 13), the decision threshold was heuristically set to `0.70`.
Because the Platt calibrator outputs highly skewed probabilities for this test set (clustering around ~0.977), evaluating at 0.70 resulted in classifying **all samples as SYNTHETIC**.

| Metric | Score |
|--------|-------|
| Accuracy | 0.7326 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.7326 |
| Recall | 1.0000 |
| Specificity | 0.0000 |
| F1 Score | 0.8457 |
| MCC | 0.0000 |
| Confusion Matrix | TN=0, FP=50, FN=0, TP=137 |

---

## 2. Final Application Decision Policy
*Status: FINAL PRODUCTION THRESHOLD*

To establish a defensible production threshold without leaking information from the locked test set, we utilized the **DEV split**.
By optimizing Youden's J statistic on the `v22_2_dev.csv` predictions, we derived the DEV-derived operating-point reference of **0.9773**.

The final application (`backend/inference.py`) strictly enforces this DEV-derived policy:
- **SYNTHETIC**: Probability ≥ 0.99
- **UNCERTAIN**: 0.8273 < Probability < 0.99
- **AUTHENTIC**: Probability ≤ 0.8273

### 3. Application Performance on Locked Test Set
*Status: PURE READ-ONLY OBSERVATION*

When the DEV-derived operating-point reference of `0.9773` is evaluated on the **frozen raw locked-test predictions**, the resulting metrics drastically shift. Since the locked-test probabilities cluster around `0.9772`, they fall entirely below the 0.9773 reference point. Consequently, under a binary evaluation, the model classifies **all samples as AUTHENTIC**.

| Metric | Score |
|--------|-------|
| Accuracy | 0.2674 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| Specificity | 1.0000 |
| F1 Score | 0.0000 |
| MCC | 0.0000 |
| Confusion Matrix | TN=50, FP=0, FN=137, TP=0 |

*(Note: Under the 3-state application policy, these probabilities technically map to the "UNCERTAIN" band (0.8273-0.99), accurately reflecting the model's inability to confidently discriminate these samples).*

## 4. Conclusion
The ROC-AUC (0.6013), PR-AUC (0.7988), and Brier Score (0.2557) remain tied to the frozen prediction/calibration definition and correctly reflect the underlying, albeit modest, signal. 

The discrepancy between the 0.70 (100% Synthetic) and 0.9773 (100% Authentic / Uncertain) operating points perfectly illustrates threshold sensitivity under the evaluated data distribution. The DEV-derived operating-point reference has been finalized in the backend and frontend to maintain strict scientific separation between development tuning and test evaluation.
