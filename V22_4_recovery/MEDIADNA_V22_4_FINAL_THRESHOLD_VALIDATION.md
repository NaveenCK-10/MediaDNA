# MediaDNA V22.4 Final Threshold Validation

## Binary DEV Benchmark (Raw Threshold 0.60)
The target operating point was empirically derived from the DEV set to balance heavily synthetic-skewed prior probabilities.
- **Balanced Accuracy**: 0.8634
- **MCC**: 0.2355
- **Precision**: 1.0000
- **Recall**: 0.7267
- **Specificity**: 1.0000
- **Confusion Matrix**: TN 50, FP 0, FN 605, TP 1609

## 3-State Calibrated Policy Evaluation
Platt scaling calibrator mapped the 0.60 raw threshold to approximately ~0.9564 calibrated probability. The following policies were evaluated exclusively on the DEV set.

### Policy: 0.93 / 0.96
- **Counts**: AUTHENTIC (419), SYNTHETIC (1592), UNCERTAIN (253)
- **Abstention Rate**: 11.17%
- **Metrics**: BACC=0.9037 | MCC=0.2741 | Precision=1.0000 | Recall=0.8073 | Specificity=1.0000
- **Confusion Matrix**: TN 39, FP 0, FN 380, TP 1592

### Policy: 0.94 / 0.96
- **Counts**: AUTHENTIC (530), SYNTHETIC (1592), UNCERTAIN (142)
- **Abstention Rate**: 6.27%
- **Metrics**: BACC=0.8831 | MCC=0.2522 | Precision=1.0000 | Recall=0.7661 | Specificity=1.0000
- **Confusion Matrix**: TN 44, FP 0, FN 486, TP 1592

### Policy: 0.94 / 0.97
- **Counts**: AUTHENTIC (530), SYNTHETIC (1573), UNCERTAIN (161)
- **Abstention Rate**: 7.11%
- **Metrics**: BACC=0.8820 | MCC=0.2518 | Precision=1.0000 | Recall=0.7640 | Specificity=1.0000
- **Confusion Matrix**: TN 44, FP 0, FN 486, TP 1573

### Policy: 0.95 / 0.96
- **Counts**: AUTHENTIC (637), SYNTHETIC (1592), UNCERTAIN (35)
- **Abstention Rate**: 1.55%
- **Metrics**: BACC=0.8650 | MCC=0.2345 | Precision=1.0000 | Recall=0.7299 | Specificity=1.0000
- **Confusion Matrix**: TN 48, FP 0, FN 589, TP 1592

### Policy: 0.95 / 0.97
- **Counts**: AUTHENTIC (637), SYNTHETIC (1573), UNCERTAIN (54)
- **Abstention Rate**: 2.39%
- **Metrics**: BACC=0.8638 | MCC=0.2341 | Precision=1.0000 | Recall=0.7276 | Specificity=1.0000
- **Confusion Matrix**: TN 48, FP 0, FN 589, TP 1573

## Conclusion and Freeze
The **0.95 / 0.96 policy** achieves a balanced accuracy (0.8650) that is incredibly close to the targeted raw 0.60 operating point (0.8634) while minimizing abstentions (1.55%). 

This proves that the calibrated three-state policy highly approximates the desired raw operating point while providing a small, 1.55% uncertainty buffer for borderline cases. The policy has been frozen to `V22_4_THRESHOLD_POLICY.json`.

**Performance Disclaimer:** 
AVFF demonstrates substantial ranking discrimination (ROC-AUC ≈ 0.91). This is NOT perfect separation. False negatives remain expected.
