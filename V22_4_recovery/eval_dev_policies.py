import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, balanced_accuracy_score,
                             confusion_matrix)
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRED_CSV = os.path.join(PROJECT_ROOT, "V22_2_BASELINE_PREDICTIONS.csv")
CALIBRATOR_PATH = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_PLATT_CALIBRATOR.pkl")

def compute_metrics(labels, preds):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "MCC": float(matthews_corrcoef(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def main():
    # Load DEV predictions
    df = pd.read_csv(PRED_CSV)
    labels = df['label'].values
    raw_probs = df['prob'].values
    
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)
        
    logits = np.log(raw_probs / (1 - raw_probs + 1e-12) + 1e-12).reshape(-1, 1)
    cal_probs = calibrator.predict_proba(logits)[:, 1]
    
    print("=== RAW BINARY BASELINE (Threshold = 0.60) ===")
    raw_preds = (raw_probs >= 0.60).astype(int)
    raw_metrics = compute_metrics(labels, raw_preds)
    for k, v in raw_metrics.items():
        print(f"{k}: {v}")
    print()
    
    policies = [
        (0.93, 0.96),
        (0.94, 0.96),
        (0.94, 0.97),
        (0.95, 0.96),
        (0.95, 0.97)
    ]
    
    results = {}
    print("=== CALIBRATED 3-STATE POLICIES ===")
    for auth, syn in policies:
        print(f"Policy: AUTHENTIC < {auth} | SYNTHETIC >= {syn}")
        decisions = []
        for p in cal_probs:
            if p >= syn:
                decisions.append("SYNTHETIC")
            elif p < auth:
                decisions.append("AUTHENTIC")
            else:
                decisions.append("UNCERTAIN")
        
        decisions = np.array(decisions)
        
        n_auth = np.sum(decisions == "AUTHENTIC")
        n_syn = np.sum(decisions == "SYNTHETIC")
        n_unc = np.sum(decisions == "UNCERTAIN")
        total = len(decisions)
        abst_rate = n_unc / total
        
        print(f"  AUTHENTIC count: {n_auth}")
        print(f"  SYNTHETIC count: {n_syn}")
        print(f"  UNCERTAIN count: {n_unc}")
        print(f"  Abstention rate: {abst_rate:.4f}")
        
        # Filter for non-uncertain
        valid_idx = decisions != "UNCERTAIN"
        if np.sum(valid_idx) > 0:
            valid_labels = labels[valid_idx]
            valid_preds = (decisions[valid_idx] == "SYNTHETIC").astype(int)
            pol_metrics = compute_metrics(valid_labels, valid_preds)
            for k, v in pol_metrics.items():
                if k != 'confusion_matrix':
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            pol_metrics = None
            print("  No confident predictions.")
        print()

if __name__ == "__main__":
    main()
