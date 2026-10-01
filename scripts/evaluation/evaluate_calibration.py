import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

def main():
    # 1. Look at V22_3_REPAIRED_TRIAD_PREDICTIONS.csv
    df = pd.read_csv("V22_3_REPAIRED_TRIAD_PREDICTIONS.csv")
    
    # Check the labels and fusion probs
    print("Dataset distribution:")
    print(df['label'].value_counts())
    
    # 2. Balance the dataset to 50/50 for calibration if necessary, 
    # but the instructions say "Use CALIBRATION split only."
    # Wait, maybe the V22_3_REPAIRED_TRIAD_PREDICTIONS.csv IS the CAL split?
    # Let's see the unique counts
    print(f"Total samples: {len(df)}")
    
    # Calculate fusion stats
    print("\nReal (label=0) stats:")
    real = df[df['label'] == 0]['repaired_fusion_prob']
    print(real.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
    
    print("\nFake (label=1) stats:")
    fake = df[df['label'] == 1]['repaired_fusion_prob']
    print(fake.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
    
    # Fit Platt Scaler
    cal_fusion = df['repaired_fusion_prob'].values.reshape(-1, 1)
    cal_labels = df['label'].values
    
    # Convert fusion_prob to logit for proper Platt scaling
    cal_fusion_logit = np.log(cal_fusion / (1 - cal_fusion))
    
    platt = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    platt.fit(cal_fusion_logit, cal_labels)
    
    print("\nBalanced Platt Scaling (on logits):")
    print(f"Coef: {platt.coef_}, Intercept: {platt.intercept_}")
    
    preds = platt.predict_proba(cal_fusion_logit)[:, 1]
    
    print(f"Calibrated probability for Real mean: {preds[cal_labels == 0].mean()}")
    print(f"Calibrated probability for Real min/max: {preds[cal_labels == 0].min()} / {preds[cal_labels == 0].max()}")
    print(f"Calibrated probability for Fake mean: {preds[cal_labels == 1].mean()}")
    print(f"Calibrated probability for Fake min/max: {preds[cal_labels == 1].min()} / {preds[cal_labels == 1].max()}")
    
    # Let's see what happens to a monotonic sequence of probabilities
    test_probs = np.linspace(0.01, 0.99, 10)
    test_logits = np.log(test_probs / (1 - test_probs)).reshape(-1, 1)
    test_calib = platt.predict_proba(test_logits)[:, 1]
    print("\nMonotonic sequence check:")
    for p, c in zip(test_probs, test_calib):
        print(f"Raw: {p:.2f} -> Calibrated: {c:.4f}")
    
    # Compare with standard (unbalanced) platt scaling
    platt_ub = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_ub.fit(cal_fusion, cal_labels)
    preds_ub = platt_ub.predict_proba(cal_fusion)[:, 1]
    
    print("\nUnbalanced Platt Scaling:")
    print(f"Coef: {platt_ub.coef_}, Intercept: {platt_ub.intercept_}")
    print(f"Calibrated probability for Real mean: {preds_ub[cal_labels == 0].mean()}")
    print(f"Calibrated probability for Fake mean: {preds_ub[cal_labels == 1].mean()}")
    
if __name__ == "__main__":
    main()
