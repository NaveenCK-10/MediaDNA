# V16.1 Phase 3 Robustness Analysis & Candidate Selection

## Candidate Performance Comparison

### Cand_A

| Noise    |   roc_auc |   pr_auc |   acc |   bal_acc |     f1 |   prec |   recall |   fpr |    fnr |   TN |   FP |   FN |   TP |   real_mean |   fake_mean |
|:---------|----------:|---------:|------:|----------:|-------:|-------:|---------:|------:|-------:|-----:|-----:|-----:|-----:|------------:|------------:|
| Clean    |    0.9145 |   0.9702 | 0.87  |    0.9    | 0.9065 | 0.9844 |   0.84   |  0.04 | 0.16   |   48 |    2 |   24 |  126 |      0.545  |      0.92   |
| Mild     |    0.9126 |   0.968  | 0.865 |    0.9033 | 0.9018 | 0.992  |   0.8267 |  0.02 | 0.1733 |   49 |    1 |   26 |  124 |      0.563  |      0.8643 |
| Moderate |    0.891  |   0.9643 | 0.83  |    0.78   | 0.8859 | 0.8919 |   0.88   |  0.32 | 0.12   |   34 |   16 |   18 |  132 |      0.5947 |      0.661  |
| Strong   |    0.7528 |   0.8985 | 0.75  |    0.5    | 0.8571 | 0.75   |   1      |  1    | 0      |    0 |   50 |    0 |  150 |      0.6665 |      0.6973 |

### Cand_B

| Noise    |   roc_auc |   pr_auc |   acc |   bal_acc |     f1 |   prec |   recall |   fpr |    fnr |   TN |   FP |   FN |   TP |   real_mean |   fake_mean |
|:---------|----------:|---------:|------:|----------:|-------:|-------:|---------:|------:|-------:|-----:|-----:|-----:|-----:|------------:|------------:|
| Clean    |    0.9216 |   0.9726 | 0.865 |    0.8167 | 0.9103 | 0.9073 |   0.9133 |  0.28 | 0.0867 |   36 |   14 |   13 |  137 |      0.5806 |      0.9316 |
| Mild     |    0.9189 |   0.9677 | 0.75  |    0.5    | 0.8571 | 0.75   |   1      |  1    | 0      |    0 |   50 |    0 |  150 |      0.699  |      0.947  |
| Moderate |    0.9124 |   0.9722 | 0.75  |    0.5    | 0.8571 | 0.75   |   1      |  1    | 0      |    0 |   50 |    0 |  150 |      0.8066 |      0.923  |
| Strong   |    0.8257 |   0.933  | 0.75  |    0.5    | 0.8571 | 0.75   |   1      |  1    | 0      |    0 |   50 |    0 |  150 |      0.884  |      0.925  |

### Cand_C

| Noise    |   roc_auc |   pr_auc |   acc |   bal_acc |     f1 |   prec |   recall |   fpr |    fnr |   TN |   FP |   FN |   TP |   real_mean |   fake_mean |
|:---------|----------:|---------:|------:|----------:|-------:|-------:|---------:|------:|-------:|-----:|-----:|-----:|-----:|------------:|------------:|
| Clean    |    0.9263 |   0.9733 | 0.865 |    0.9033 | 0.9018 | 0.992  |   0.8267 |  0.02 | 0.1733 |   49 |    1 |   26 |  124 |      0.4792 |      0.898  |
| Mild     |    0.9161 |   0.9717 | 0.69  |    0.7867 | 0.7417 | 0.9889 |   0.5933 |  0.02 | 0.4067 |   49 |    1 |   61 |   89 |      0.4822 |      0.7085 |
| Moderate |    0.8751 |   0.9552 | 0.325 |    0.55   | 0.1818 | 1      |   0.1    |  0    | 0.9    |   50 |    0 |  135 |   15 |      0.4941 |      0.535  |
| Strong   |    0.7679 |   0.9025 | 0.43  |    0.6    | 0.4062 | 0.9286 |   0.26   |  0.06 | 0.74   |   47 |    3 |  111 |   39 |      0.5474 |      0.5825 |

### Cand_D

| Noise    |   roc_auc |   pr_auc |   acc |   bal_acc |     f1 |   prec |   recall |   fpr |    fnr |   TN |   FP |   FN |   TP |   real_mean |   fake_mean |
|:---------|----------:|---------:|------:|----------:|-------:|-------:|---------:|------:|-------:|-----:|-----:|-----:|-----:|------------:|------------:|
| Clean    |    0.8949 |   0.9536 | 0.86  |    0.8133 | 0.9067 | 0.9067 |   0.9067 |  0.28 | 0.0933 |   36 |   14 |   14 |  136 |      0.6245 |      0.943  |
| Mild     |    0.9223 |   0.973  | 0.775 |    0.5833 | 0.8657 | 0.7838 |   0.9667 |  0.8  | 0.0333 |   10 |   40 |    5 |  145 |      0.6987 |      0.9565 |
| Moderate |    0.8966 |   0.9669 | 0.75  |    0.5    | 0.8571 | 0.75   |   1      |  1    | 0      |    0 |   50 |    0 |  150 |      0.9126 |      0.983  |
| Strong   |    0.8213 |   0.9369 | 0.75  |    0.5    | 0.8571 | 0.75   |   1      |  1    | 0      |    0 |   50 |    0 |  150 |      0.9775 |      0.9893 |

## Robustness Analysis

### A. Clean Performance
- **Cand_A_Clean (V15.4 Baseline)**: ROC-AUC = 0.9145, Accuracy = 0.8700, FPR = 0.0400, FNR = 0.1600.
- **Cand_B_Mild**: Slightly better clean ROC-AUC (0.9216) but worse FPR (0.2800) and much worse Balanced Accuracy (0.8167).
- **Cand_C_Moderate**: Better clean ROC-AUC (0.9263), lower FPR (0.0200). Accuracy is slightly lower (0.8650) with similar Balanced Acc (0.9033).
- **Cand_D_Mixed**: Clean ROC-AUC is lower (0.8949). Accuracy is 0.8600. FPR is extremely high (0.2800).

### B. Mild-Noise Performance
- **Cand_A_Clean**: ROC-AUC = 0.9126. FPR drops to 0.0200.
- **Cand_B_Mild**: ROC-AUC = 0.9189. FPR explodes to 1.0000. Catastrophic real-video failure.
- **Cand_C_Moderate**: ROC-AUC = 0.9161. FPR is excellent (0.0200). Accuracy drops significantly due to FNR = 0.4067.
- **Cand_D_Mixed**: ROC-AUC = 0.9223. FPR explodes to 0.8000. Catastrophic real-video failure.

### C. Moderate-Noise Performance
- **Cand_A_Clean**: ROC-AUC = 0.8910. FPR spikes to 0.3200.
- **Cand_B_Mild**: ROC-AUC = 0.9124. FPR is 1.0000.
- **Cand_C_Moderate**: ROC-AUC = 0.8751. FPR is 0.0000 (perfect real retention), but FNR spikes to 0.9000.
- **Cand_D_Mixed**: ROC-AUC = 0.8966. FPR is 1.0000.

### D. Strong-Noise Performance
- **Cand_A_Clean**: ROC-AUC = 0.7528. FPR = 1.0000.
- **Cand_B_Mild**: ROC-AUC = 0.8257. FPR = 1.0000.
- **Cand_C_Moderate**: ROC-AUC = 0.7679. FPR = 0.0600. FNR is very high (0.7400).
- **Cand_D_Mixed**: ROC-AUC = 0.8213. FPR = 1.0000.

### E. Conclusion on Degradation & False Positives (FPR)
The V15.4 Baseline (Cand A) degrades under noise by misclassifying real videos as fake (FPR spikes to 0.3200 at Moderate, 1.0000 at Strong).
Cand B (Mild) and Cand D (Mixed) are utterly destroyed by False Positives; their FPR hits 1.0000 even at Moderate noise. They simply predict everything is fake.
Cand C (Moderate) exhibits a totally different failure mode: its FPR remains pristine under noise (0.02 -> 0.00 -> 0.06), but it fails by missing fakes (FNR spikes to 0.9000). While failing safe is better than flagging reals, the FNR is too high.

## Training Comparability
All models were trained under identical configuration except for noise augmentation, using the verified batch_size=1 setting to avoid OOM.

## Locked Test Gate Assessment
**Does any candidate clearly deserve locked-test evaluation?**

**NO.**

None of the candidates demonstrate a meaningful and credible improvement over the V15.4 baseline. 
- Candidates B and D suffer catastrophic False Positive Rates on real videos under noise (FPR = 1.0). 
- Candidate C preserves FPR nicely under noise but its True Positive Rate completely collapses (FNR = 0.90). 
- Candidate A (the existing V15.4 baseline) remains the most balanced model.

**Recommendation:** Do NOT run the locked test. V15.4 remains the production baseline.
