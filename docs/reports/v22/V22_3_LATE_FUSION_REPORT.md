# V22.3 Strict Late Fusion Report

**Date:** 2026-09-28T18:06:35.264508
**Method:** simple_weighted_average
**Selected Config:** equal_average
**Weights:** Visual=0.5, Audio=0.5

## DEV Metrics (threshold=0.5)

| Metric | Value |
|--------|-------|
| ROC_AUC | 0.6315 |
| PR_AUC | 0.9885 |
| accuracy | 0.9813 |
| balanced_accuracy | 0.5000 |
| precision | 0.9813 |
| recall | 1.0000 |
| F1 | 0.9906 |
| MCC | 0.0000 |
| specificity | 0.0000 |

## DEV Metrics (optimal threshold=0.79)

| Metric | Value |
|--------|-------|
| ROC_AUC | 0.6315 |
| PR_AUC | 0.9885 |
| accuracy | 0.9778 |
| balanced_accuracy | 0.5115 |
| precision | 0.9818 |
| recall | 0.9959 |
| F1 | 0.9888 |
| MCC | 0.0461 |
| specificity | 0.0270 |

## Quadrant Analysis

### RVRA
- N: 37
- Accuracy: 0.0000
- Mean fusion score: 0.8194
- Median fusion score: 0.8220

### RVFA
- N: 37
- Accuracy: 1.0000
- Mean fusion score: 0.8196
- Median fusion score: 0.8221

### FVRA
- N: 833
- Accuracy: 1.0000
- Mean fusion score: 0.8205
- Median fusion score: 0.8228

### FVFA
- N: 1075
- Accuracy: 1.0000
- Mean fusion score: 0.8220
- Median fusion score: 0.8243

