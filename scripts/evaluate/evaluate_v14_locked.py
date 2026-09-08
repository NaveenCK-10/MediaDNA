"""
V14 Locked Test Evaluation Script.

After training is COMPLETE and FROZEN, this script evaluates the V14 checkpoint
against the locked test set ONCE. No further tuning is permitted after this.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score
)

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService

def evaluate_locked_test():
    # Try V14, fall back to V8
    v14_ckpt = r"checkpoints\v14_fullscale\models\best_audio_model.pth"
    v8_ckpt = r"checkpoints\v8\baseline\models\best_audio_model.pth"
    
    checkpoint_path = v14_ckpt if os.path.exists(v14_ckpt) else v8_ckpt
    ckpt_name = "V14" if os.path.exists(v14_ckpt) else "V8"
    
    # Use V14's own locked test set if it exists, else use V8's
    v14_test = r"data\test_locked_v14.csv"
    v8_test = r"data\test_locked_v8.csv"
    csv_path = v14_test if os.path.exists(v14_test) else v8_test
    test_name = "V14" if os.path.exists(v14_test) else "V8"
    
    print(f"Checkpoint: {checkpoint_path} ({ckpt_name})")
    print(f"Test Set: {csv_path} ({test_name})")
    
    service = OpenAVFFService(checkpoint_path=checkpoint_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} videos.")
    
    y_true, y_pred, y_prob, types = [], [], [], []
    cat_results = {
        "RealVideo-RealAudio": {"true": [], "pred": [], "prob": []},
        "RealVideo-FakeAudio": {"true": [], "pred": [], "prob": []},
        "FakeVideo-RealAudio": {"true": [], "pred": [], "prob": []},
        "FakeVideo-FakeAudio": {"true": [], "pred": [], "prob": []}
    }
    
    errors = []
    
    for idx, row in df.iterrows():
        video_path = os.path.join(os.path.dirname(__file__), row['video_path'])
        label = int(row['label'])
        v_type = row['type']
        
        try:
            res = service.analyze_video(video_path)
            prob = res.fake_probability
            pred = 1 if prob >= 0.5 else 0
            
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)
            types.append(v_type)
            
            cat_results[v_type]["true"].append(label)
            cat_results[v_type]["pred"].append(pred)
            cat_results[v_type]["prob"].append(prob)
            
            correct = pred == label
            if not correct:
                errors.append({
                    "video": row['video_path'],
                    "type": v_type,
                    "true_label": label,
                    "predicted": pred,
                    "fake_prob": prob,
                    "confidence": abs(prob - 0.5)
                })
            
            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(df)} videos...")
        except Exception as e:
            print(f"Error: {video_path}: {e}")
            
    # --- GLOBAL METRICS ---
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    map_score = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f"V14 LOCKED TEST RESULTS (Checkpoint: {ckpt_name}, Test: {test_name})")
    print("="*60)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {b_acc:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print(f"mAP:               {map_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("\n--- CATEGORY METRICS ---")
    for cat, data in cat_results.items():
        if len(data["true"]) > 0:
            c_acc = accuracy_score(data["true"], data["pred"])
            c_prec = precision_score(data["true"], data["pred"], zero_division=0)
            c_rec = recall_score(data["true"], data["pred"], zero_division=0)
            c_f1 = f1_score(data["true"], data["pred"], zero_division=0)
            print(f"{cat} (N={len(data['true'])}): Acc={c_acc:.4f} Prec={c_prec:.4f} Rec={c_rec:.4f} F1={c_f1:.4f}")
    
    # --- ERROR ANALYSIS ---
    print(f"\n--- ERROR ANALYSIS ({len(errors)} total errors) ---")
    errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)
    print("Top 10 most confident errors:")
    for i, e in enumerate(errors_sorted[:10]):
        print(f"  {i+1}. [{e['type']}] true={e['true_label']} pred={e['predicted']} prob={e['fake_prob']:.4f} file={os.path.basename(e['video'])}")
    
    # Save raw results
    os.makedirs("reports", exist_ok=True)
    out_df = pd.DataFrame({"true": y_true, "pred": y_pred, "prob": y_prob, "type": types})
    out_df.to_csv("reports/v14_locked_test_results.csv", index=False)
    
    # Save error analysis
    err_df = pd.DataFrame(errors_sorted)
    err_df.to_csv("reports/v14_error_analysis.csv", index=False)
    print(f"\nSaved results to reports/v14_locked_test_results.csv")
    print(f"Saved error analysis to reports/v14_error_analysis.csv")

if __name__ == "__main__":
    evaluate_locked_test()
