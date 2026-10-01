# MediaDNA Final Results

**Date:** 2026-09-28
**Checkpoint Version:** V22.3 (Visual Specialist B, Audio Specialist C)
**Fusion Policy:** Strict Late Fusion (0.5 Visual + 0.5 Audio)
**Calibration Method:** Platt Scaling (fitted on CAL split)
**DEV-derived operating-point reference:** 0.9773 (from DEV Youden's J Statistic)

---

## 1. DEV Split Performance (Pre-Thresholding)
*Split: `v22_2_dev.csv` (2,264 samples)*

| Model Component | ROC-AUC | PR-AUC |
|-----------------|---------|--------|
| Original V14 AVFF Baseline | 0.9022 | 0.9976 |
| V22.3B Visual Specialist | 0.5335 | 0.9815 |
| V22.3C Audio Specialist | 0.7408 | 0.9916 |
| **V22.3 Repaired Late Fusion** | **0.6182** | **0.9830** |

*Note: The Original V14 Baseline suffered from catastrophic pooling collapse (zeroing audio) but benefited from extreme dataset leakage. The Repaired Late Fusion represents true, unbiased performance.*

---

## 2. LOCKED TEST Generalization
*Split: `v22_2_test.csv` (187 Samples: 137 Fake, 50 Real)*
*Evaluated on Calibrated Probability at the DEV-derived operating-point reference of 0.9773*

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.6013 |
| **PR-AUC** | 0.7988 |
| **Brier Score** | 0.2557 |
| **Accuracy** | 0.7433 |
| **Balanced Accuracy** | 0.5200 |
| **Precision** | 0.7380 |
| **Recall** | 1.0000 |
| **Specificity** | 0.0400 |
| **F1 Score** | 0.8483 |
| **MCC** | 0.1250 |

### Interpretation
At the DEV-derived operating-point reference of `0.9773`, the pipeline predicts almost all test samples as SYNTHETIC. While the model correctly identifies all Fakes (Recall 1.0), it struggles heavily with False Positives on pristine data (Specificity 0.04). The underlying ROC-AUC (0.6013) demonstrates modest generalized signal, but the model requires further adaptation for balanced production use.
