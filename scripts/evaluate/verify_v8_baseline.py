import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, balanced_accuracy_score

# Ensure the backend module is accessible
sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService

def evaluate_v8():
    checkpoint_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\checkpoints\v8\baseline\models\best_audio_model.pth"
    csv_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\test_locked_v8.csv"
    
    # If the local path doesn't exist, try the medium path as fallback per some old scripts
    if not os.path.exists(checkpoint_path):
        checkpoint_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-local\models\best_audio_model.pth"
        
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
        
    if not os.path.exists(csv_path):
        print(f"ERROR: Test CSV not found at {csv_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    service = OpenAVFFService(checkpoint_path=checkpoint_path)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} videos from {csv_path}")
    
    y_true = []
    y_pred = []
    y_prob = []
    types = []
    
    # Track metrics per category
    cat_results = {
        "RealVideo-RealAudio": {"true": [], "pred": [], "prob": []},
        "RealVideo-FakeAudio": {"true": [], "pred": [], "prob": []},
        "FakeVideo-RealAudio": {"true": [], "pred": [], "prob": []},
        "FakeVideo-FakeAudio": {"true": [], "pred": [], "prob": []}
    }
    
    total = len(df)
    for idx, row in df.iterrows():
        video_path = os.path.join(r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF", row['video_path'])
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
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total} videos...")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            
    print("\n--- GLOBAL METRICS ---")
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    map_score = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {b_acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"mAP: {map_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("\n--- CATEGORY METRICS ---")
    for cat, data in cat_results.items():
        if len(data["true"]) > 0:
            c_acc = accuracy_score(data["true"], data["pred"])
            
            # Additional category metrics if applicable
            try:
                c_auc = roc_auc_score(data["true"], data["prob"]) if len(set(data["true"])) > 1 else float('nan')
            except:
                c_auc = float('nan')
                
            print(f"{cat} (N={len(data['true'])}): Accuracy = {c_acc:.4f} | ROC-AUC = {c_auc:.4f}")
            
    # Save raw results
    out_df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "prob": y_prob,
        "type": types
    })
    os.makedirs(os.path.join(os.path.dirname(__file__), "reports"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "reports", "v8_reverification.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved raw predictions to {out_path}")

if __name__ == "__main__":
    evaluate_v8()
