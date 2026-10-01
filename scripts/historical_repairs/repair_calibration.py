import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline


def main():
    print("=== REPAIRING PLATT CALIBRATION ===")
    
    df = pd.read_csv("V22_3_REPAIRED_TRIAD_PREDICTIONS.csv")
    
    cal_fusion = df['repaired_fusion_prob'].values.reshape(-1, 1)
    cal_labels = df['label'].values
    
    cal_logits = np.log(cal_fusion / (1 - cal_fusion + 1e-12))
    
    # Use class_weight='balanced' so the model doesn't just predict the 98% synthetic dataset prior!
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    lr.fit(cal_logits, cal_labels)
    calibrator = lr
    
    print("Calibration fit complete.")
    
    # Let's run a monotonic sequence check
    test_probs = np.linspace(0.01, 0.99, 10).reshape(-1, 1)
    test_logits = np.log(test_probs / (1 - test_probs + 1e-12))
    test_calib = calibrator.predict_proba(test_logits)[:, 1]
    
    print("\\nMonotonic Sequence Sanity Check:")
    for p, c in zip(test_probs.flatten(), test_calib):
        print(f"Raw Score: {p:.2f} -> Calibrated Prob: {c:.4f}")
        
    # Save the repaired calibrator over the old one
    with open("V22_3_PLATT_CALIBRATOR.pkl", "wb") as f:
        pickle.dump(calibrator, f)
        
    print("\nRepaired calibrator saved to V22_3_PLATT_CALIBRATOR.pkl")
    
    # Print out summary statistics for the decision policy
    preds = calibrator.predict_proba(cal_logits)[:, 1]
    real_mean = preds[cal_labels == 0].mean()
    fake_mean = preds[cal_labels == 1].mean()
    print(f"\nReal Samples (Mean Calibrated Prob): {real_mean:.4f}")
    print(f"Synthetic Samples (Mean Calibrated Prob): {fake_mean:.4f}")
    
if __name__ == "__main__":
    main()
