import os
import hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix,
    brier_score_loss, f1_score
)
import pickle
import time
import sys

# Add backend to path to import inference
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from inference import OpenAVFFService

def get_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    print("=== 1. INDIVIDUAL VIDEO EVALUATION ===")
    
    print("Loading V22.3 Pipeline...")
    service = OpenAVFFService(checkpoint_path="C:\\Users\\navee\\Desktop\\Projects\\MediaDna\\OpenAVFF\\checkpoints\\v14_fullscale\\models\\best_audio_model.pth")
    print("Models loaded successfully.")
    
    videos = [
        ("Exact Video (Create_a_photorealistic_AI_gen.mp4)", r"C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4", 1), 
        ("Known Real (00143_clean.mp4)", r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00143_clean.mp4", 0),
        ("Known Synthetic (00160_id01098_wavtolip_clean.mp4)", r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00160_id01098_wavtolip_clean.mp4", 1)
    ]
    
    for name, path, label_true in videos:
        print(f"\n[{name}]")
        if not os.path.exists(path):
            print("Status: FILE_NOT_FOUND")
            continue
            
        print(f"SHA-256: {get_sha256(path)}")
        try:
            result = service.analyze_video(path, progress_cb=None, context={})
            
            v = result.get("visual", {}).get("anomaly_score")
            a = result.get("audio", {}).get("anomaly_score")
            f = result.get("multimodal", {}).get("fusion_output")
            c = result.get("trust", {}).get("calibrated_probability")
            d = result.get("trust", {}).get("abstention_state")
            
            is_correct = "INCONCLUSIVE"
            if d == "AUTHENTIC" and label_true == 0: is_correct = "CORRECT"
            elif d == "AUTHENTIC" and label_true == 1: is_correct = "INCORRECT"
            elif d == "SYNTHETIC" and label_true == 1: is_correct = "CORRECT"
            elif d == "SYNTHETIC" and label_true == 0: is_correct = "INCORRECT"
            
            print("Execution: PASS")
            print(f"Classification: {is_correct}")
            print(f"Visual score: {v}")
            print(f"Audio score: {a}")
            print(f"Fusion score: {f}")
            print(f"Calibrated output: {c}")
            print(f"Final decision: {d}")
        except Exception as e:
            print(f"Execution: FAIL")
            print(f"Error: {e}")

    print("\n=== 2. COMPLETE CAL/DEV EVALUATION ===")
    df = pd.read_csv("V22_3_REPAIRED_TRIAD_PREDICTIONS.csv")
    y_true = df['label'].values
    y_raw = df['repaired_fusion_prob'].values
    
    roc_auc = roc_auc_score(y_true, y_raw)
    precision, recall, _ = precision_recall_curve(y_true, y_raw)
    pr_auc = auc(recall, precision)
    
    # Evaluate hard metrics using raw >= 0.5 (as asked for raw detector evaluation)
    y_pred_raw = (y_raw >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred_raw)
    mcc = matthews_corrcoef(y_true, y_pred_raw)
    prec = precision_score(y_true, y_pred_raw, zero_division=0)
    rec = recall_score(y_true, y_pred_raw)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_raw).ravel()
    specificity = tn / (tn + fp + 1e-9)
    f1 = f1_score(y_true, y_pred_raw)
    
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("Balanced Accuracy (raw >= 0.5):", bal_acc)
    print("MCC:", mcc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("Specificity:", specificity)
    print("F1:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_raw))
    
    print("\n=== 3. SCORE STATISTICS ===")
    real_scores = y_raw[y_true == 0]
    syn_scores = y_raw[y_true == 1]
    
    print("REAL:")
    print(f"n = {len(real_scores)}")
    print(f"mean raw score = {np.mean(real_scores)}")
    print(f"std = {np.std(real_scores)}")
    print(f"median = {np.median(real_scores)}")
    print(f"p25 = {np.percentile(real_scores, 25)}")
    print(f"p75 = {np.percentile(real_scores, 75)}")
    
    print("\nSYNTHETIC:")
    print(f"n = {len(syn_scores)}")
    print(f"mean raw score = {np.mean(syn_scores)}")
    print(f"std = {np.std(syn_scores)}")
    print(f"median = {np.median(syn_scores)}")
    print(f"p25 = {np.percentile(syn_scores, 25)}")
    print(f"p75 = {np.percentile(syn_scores, 75)}")
    
    print("\n=== 4. CALIBRATION EVALUATION ===")
    def compute_ece(y_t, y_p, n_bins=10):
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_p, bins) - 1
        bin_sums = np.bincount(binids, weights=y_p, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_t, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_t)))
        return ece

    with open("V22_3_PLATT_CALIBRATOR.pkl", "rb") as f:
        calibrator = pickle.load(f)
    
    y_raw_logits = np.log(y_raw / (1 - y_raw + 1e-12))
    y_calib = calibrator.predict_proba(y_raw_logits.reshape(-1, 1))[:, 1]
    
    brier_before = brier_score_loss(y_true, y_raw)
    brier_after = brier_score_loss(y_true, y_calib)
    
    ece_before = compute_ece(y_true, y_raw)
    ece_after = compute_ece(y_true, y_calib)
    
    print(f"Brier score before calibration: {brier_before}")
    print(f"Brier score after calibration: {brier_after}")
    print(f"ECE before calibration: {ece_before}")
    print(f"ECE after calibration: {ece_after}")

if __name__ == "__main__":
    main()
