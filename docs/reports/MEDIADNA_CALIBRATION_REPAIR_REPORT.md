# MEDIADNA CALIBRATION REPAIR REPORT

## 1. Issue: Monotonic Collapse
During the Phase 3 and Phase 4 audit, a critical calibration failure was identified. The previous active calibrator artifact `V22_3_PLATT_CALIBRATOR.pkl` was mathematically collapsed. 
The coefficients forced any realistic raw fusion score to map asymptotically to ~0.97728. Because the `AUTHENTIC` and `SYNTHETIC` thresholds were 0.8273 and 0.99 respectively, the vast majority of media, both real and synthetic, evaluated to a calibrated probability between 0.8273 and 0.99, forcing them into the `UNCERTAIN` bucket, or if they marginally touched 0.99, misclassified known REAL as `SYNTHETIC`.

## 2. Investigation
By extracting the exact raw fusion scores for the V22.3 model across the validation triad, we discovered that the raw model scores naturally fall tightly in the `0.80 - 0.83` range with very poor class separability (mean Real: ~0.82, mean Synthetic: ~0.82).

The original calibration attempted to scale this using a standard `LogisticRegression` fitted on raw probabilities. Due to the tight clustering and heavy imbalance (98% synthetic), the regression model simply collapsed to predicting the prior.

## 3. The Repair
The calibration was completely rebuilt (`repair_calibration.py`) using a sound mathematical approach:
1. Transform raw bounded probability scores `[0,1]` to unbounded logit space: `np.log(P / (1 - P + 1e-12))`
2. Fit a `LogisticRegression` on the logits using `class_weight='balanced'` to offset the synthetic bias.
3. Save the resulting unbiased LR model as the new `V22_3_PLATT_CALIBRATOR.pkl`.

## 4. Repaired Outcome
The repaired calibration successfully produces a monotonic sequence across the `0.01 - 0.99` spectrum without collapsing.

**Sanity Check Output:**
```
Raw Score: 0.01 -> Calibrated Prob: 0.0000
Raw Score: 0.45 -> Calibrated Prob: 0.0188
Raw Score: 0.55 -> Calibrated Prob: 0.0492
Raw Score: 0.77 -> Calibrated Prob: 0.3350
Raw Score: 0.99 -> Calibrated Prob: 0.9991
```

## 5. Scientific Limitation Exposed
With an unbiased, functional calibrator, the *true* separability of the V22.3 pipeline was exposed. On the validation set:
- Real Samples (Mean Calibrated Prob): 0.4977
- Synthetic Samples (Mean Calibrated Prob): 0.5022

The model genuinely lacks the necessary statistical evidence to differentiate the current dataset. Consequently, we restored the honest 3-state decision policy derived from this calibration:
- AUTHENTIC: < 0.30
- UNCERTAIN: 0.30 - 0.70
- SYNTHETIC: >= 0.70

Because the model averages ~0.50 for all media, it will honestly output `UNCERTAIN` for most files. This is not a bug; this is the scientifically honest expression of the model's current capability.
