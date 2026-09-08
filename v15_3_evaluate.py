import os
import sys
import csv
import random
from typing import List
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score

# Add project root
sys.path.insert(0, os.path.abspath("."))
from backend.inference import OpenAVFFService

def load_csv(csv_path: str) -> List[dict]:
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            data.append({"video": row[0], "label": int(row[1])})
    return data

def run_evaluation(service: OpenAVFFService, dataset: List[dict], desc: str):
    print(f"\n==================================================")
    print(f"EVALUATING: {desc}")
    print(f"==================================================")
    
    y_true = []
    y_pred = []
    y_scores = []
    
    # Track statistics
    for i, item in enumerate(dataset):
        vid_path = item["video"]
        label = item["label"] # 1=Fake, 0=Real
        
        try:
            res = service.analyze_video(vid_path)
            prob = res.fake_probability
            
            y_true.append(label)
            y_scores.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
            
            # Print periodic progress
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(dataset)}... (Latest P(Fake): {prob:.4f}, True: {label})")
        except Exception as e:
            print(f"Skipping {vid_path}: {e}")
            
    if not y_true:
        print("No valid samples evaluated.")
        return
        
    roc_auc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"Total Evaluated: {len(y_true)}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\n--- THRESHOLD SWEEP ---")
    for thresh in [0.50, 0.55, 0.57, 0.60, 0.65, 0.70]:
        t_pred = [1 if s >= thresh else 0 for s in y_scores]
        t_acc = accuracy_score(y_true, t_pred)
        t_bal = balanced_accuracy_score(y_true, t_pred)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_true, t_pred, labels=[0, 1]).ravel()
        t_fpr = t_fp / (t_fp + t_tn) if (t_fp + t_tn) > 0 else 0
        t_fnr = t_fn / (t_fn + t_tp) if (t_fn + t_tp) > 0 else 0
        print(f"Thresh {thresh:.2f} | Acc {t_acc:.4f} | BalAcc {t_bal:.4f} | FPR {t_fpr:.4f} ({t_fp}/{t_fp+t_tn}) | FNR {t_fnr:.4f} ({t_fn}/{t_fn+t_tp})")
    
    print("\n--- DEFAULT METRICS (Thresh 0.5) ---")
    print(f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}")
    print(f"F1 Score: {f1:.4f} | Precision: {prec:.4f} | Recall (Sensitivity): {rec:.4f}")
    print(f"FPR (False Positives/Real): {fpr:.4f} ({fp}/{fp+tn})")
    print(f"FNR (False Negatives/Fake): {fnr:.4f} ({fn}/{fn+tp})")

    # Score distributions
    real_scores = [s for s, l in zip(y_scores, y_true) if l == 0]
    fake_scores = [s for s, l in zip(y_scores, y_true) if l == 1]
    
    if real_scores:
        print(f"\nReal Score Dist: Mean {sum(real_scores)/len(real_scores):.4f} | Max {max(real_scores):.4f}")
    if fake_scores:
        print(f"Fake Score Dist: Mean {sum(fake_scores)/len(fake_scores):.4f} | Min {min(fake_scores):.4f}")


def main():
    print("Loading V14 (Current Production) Model with FIXED Preprocessing...")
    service = OpenAVFFService()
    
    # 1. Real Media Regression Suite
    # We will construct a small manual list of real videos, including naveen.mp4
    regression_videos = [
        {"video": "src/naveen.mp4", "label": 0}
    ]
    # Let's pull some real videos from the valset to increase the suite size
    val_data = load_csv("data/val_v14.csv")
    real_val = [d for d in val_data if d["label"] == 0]
    fake_val = [d for d in val_data if d["label"] == 1]
    
    regression_videos.extend(real_val[:10]) # add 10 real validation videos
    
    print("\nRunning naveen.mp4 specific check:")
    res = service.analyze_video("src/naveen.mp4")
    print(f"naveen.mp4 -> Prediction: {res.prediction}, P(Fake): {res.fake_probability:.4f}")
    
    run_evaluation(service, regression_videos, "REAL MEDIA REGRESSION SUITE (11 Genuine Videos)")
    run_evaluation(service, fake_val[:20], "FAKE CONTROL GROUP (20 Manipulated Videos)")
    
    print("\n==================================================")
    print("PHASE 9: LOCKED TEST (test_locked_v14.csv)")
    print("==================================================")
    test_data = load_csv("data/test_locked_v14.csv")
    run_evaluation(service, test_data, "LOCKED V14 TEST SET (Fixed Preprocessing, Threshold Sweep)")
    
if __name__ == "__main__":
    main()
