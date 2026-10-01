import os
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix,
    brier_score_loss, f1_score
)
import pickle
import time

BASE_URL = "http://localhost:8000"

def get_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def analyze_video(name, path, label_true):
    if not os.path.exists(path):
        return {"name": name, "status": "FILE_NOT_FOUND"}
    
    sha256 = get_sha256(path)
    
    try:
        with open(path, "rb") as f:
            res = requests.post(f"{BASE_URL}/api/analyze", files={"video": f})
        
        if res.status_code != 200:
            return {"name": name, "status": "API_ERROR", "error": res.text}
            
        job_id = res.json()["job_id"]
        
        while True:
            res = requests.get(f"{BASE_URL}/api/history")
            history = res.json()
            job = next((j for j in history if j.get("id") == job_id), None)
            if job and job.get("status") in ["COMPLETED", "FAILED"]:
                break
            time.sleep(1)
            
        if job.get("status") == "COMPLETED":
            v = job.get("visual", {}).get("raw_model_score")
            a = job.get("audio", {}).get("raw_model_score")
            f = job.get("fusion", {}).get("raw_model_score")
            c = job.get("trust", {}).get("calibrated_probability")
            d = job.get("trust", {}).get("abstention_state")
            
            # evaluate correctness
            # True label: 0 for real, 1 for synthetic
            is_correct = "INCONCLUSIVE"
            if d == "AUTHENTIC" and label_true == 0: is_correct = "CORRECT"
            elif d == "AUTHENTIC" and label_true == 1: is_correct = "INCORRECT"
            elif d == "SYNTHETIC" and label_true == 1: is_correct = "CORRECT"
            elif d == "SYNTHETIC" and label_true == 0: is_correct = "INCORRECT"
            
            return {
                "name": name,
                "sha256": sha256,
                "status": "COMPLETED",
                "execution": "PASS",
                "v": v, "a": a, "f": f, "c": c, "d": d,
                "label": label_true,
                "correctness": is_correct
            }
        else:
            return {"name": name, "status": "FAILED", "execution": "FAIL", "error": job.get("error_message")}
            
    except Exception as e:
        return {"name": name, "status": "EXCEPTION", "execution": "FAIL", "error": str(e)}

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece

def main():
    print("=== 1. INDIVIDUAL VIDEO EVALUATION ===")
    videos = [
        ("Exact Video (Create_a_photorealistic_AI_gen.mp4)", r"C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4", 1), # Assume synthetic based on name
        ("Known Real (00143_clean.mp4)", r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00143_clean.mp4", 0),
        ("Known Synthetic (00160_id01098_wavtolip_clean.mp4)", r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00160_id01098_wavtolip_clean.mp4", 1)
    ]
    
    for name, path, label in videos:
        res = analyze_video(name, path, label)
        print(f"\n[{name}]")
        if res.get("sha256"): print(f"SHA-256: {res['sha256']}")
        print(f"Execution: {res.get('execution', 'FAIL')}")
        if res.get("status") == "COMPLETED":
            print(f"Classification: {res['correctness']}")
            print(f"Visual score: {res['v']}")
            print(f"Audio score: {res['a']}")
            print(f"Fusion score: {res['f']}")
            print(f"Calibrated output: {res['c']}")
            print(f"Final decision: {res['d']}")
        else:
            print(f"Error: {res.get('error')}")

    print("\n=== 2. COMPLETE CAL/DEV EVALUATION ===")
    df = pd.read_csv("V22_3_REPAIRED_TRIAD_PREDICTIONS.csv")
    y_true = df['label'].values
    y_raw = df['repaired_fusion_prob'].values
    
    # metrics using raw fusion score for ranking
    roc_auc = roc_auc_score(y_true, y_raw)
    precision, recall, _ = precision_recall_curve(y_true, y_raw)
    pr_auc = auc(recall, precision)
    
    # threshold at 0.5 on raw for hard metrics, or use calibrated labels?
    # The prompt asks to evaluate the CURRENT V22.3 raw detector
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
    with open("V22_3_PLATT_CALIBRATOR.pkl", "rb") as f:
        calibrator = pickle.load(f)
    
    # Get calibrated probabilities for the dataset
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
