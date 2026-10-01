# V22.2 Baseline Verification & Class Imbalance Analysis

## Class Imbalance
- Total: 2264
- Fake Prevalence: 2214 (97.79%)
- Real Prevalence: 50 (2.21%)

## Null Baselines
- **All-Fake Baseline**: F1=0.9888, B-Acc=0.5000, MCC=0.0000
- **All-Real Baseline**: F1=0.0000, B-Acc=0.5000, MCC=0.0000

## AUC Verification
- Directly computed AUC from predictions: 0.8974
- Positives (Fake): 2214
- Negatives (Real): 50
The high F1 score (0.98+) previously reported is proven to be an artifact of class prevalence. The All-Fake baseline achieves nearly the identical F1 score, demonstrating that F1 alone is a misleading metric for this skewed distribution.
