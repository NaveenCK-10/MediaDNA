import os
import sys
import time
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService

def get_test_videos():
    real_dir = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\RealVideo-RealAudio"
    fake_dir = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\FakeVideo-FakeAudio"

    videos = []
    
    real_count = 0
    for root, _, files in os.walk(real_dir):
        for file in files:
            if file.endswith(".mp4"):
                videos.append({"path": os.path.join(root, file), "label": 0, "expected": "real"})
                real_count += 1
                if real_count >= 20:
                    break
        if real_count >= 20:
            break
            
    fake_count = 0
    for root, _, files in os.walk(fake_dir):
        for file in files:
            if file.endswith(".mp4"):
                videos.append({"path": os.path.join(root, file), "label": 1, "expected": "fake"})
                fake_count += 1
                if fake_count >= 20:
                    break
        if fake_count >= 20:
            break
            
    return videos

def evaluate_checkpoint(ckpt_path, videos):
    print(f"\nEvaluating: {ckpt_path}")
    service = OpenAVFFService(checkpoint_path=ckpt_path)
    
    y_true = []
    y_pred = []
    y_prob = []
    
    results = []
    
    for v in videos:
        try:
            res = service.analyze_video(v["path"])
            
            y_true.append(v["label"])
            pred_label = 1 if res.fake_probability >= 0.5 else 0
            y_pred.append(pred_label)
            y_prob.append(res.fake_probability)
            
            results.append({
                "checkpoint": os.path.basename(os.path.dirname(os.path.dirname(ckpt_path))),
                "video": os.path.basename(v["path"]),
                "expected": v["expected"],
                "prediction": res.prediction,
                "fake_prob": res.fake_probability,
                "real_prob": res.real_probability
            })
        except Exception as e:
            print(f"Error on {v['path']}: {e}")
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        "checkpoint": os.path.basename(os.path.dirname(os.path.dirname(ckpt_path))),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc,
        "Fake Recall": fake_recall,
        "Real Recall": real_recall,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    }
    
    return metrics, results

def main():
    os.makedirs("reports", exist_ok=True)
    
    videos = get_test_videos()
    print(f"Loaded {len(videos)} videos for testing.")
    
    checkpoints = [
        r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-local\models\best_audio_model.pth",
        r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-medium\models\best_audio_model.pth"
    ]
    
    all_metrics = []
    all_results = []
    
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            metrics, results = evaluate_checkpoint(ckpt, videos)
            all_metrics.append(metrics)
            all_results.extend(results)
        else:
            print(f"Checkpoint not found: {ckpt}")
            
    # Write metrics
    if all_metrics:
        with open(os.path.join("reports", "checkpoint_comparison.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
            
    # Write results
    if all_results:
        with open(os.path.join("reports", "test_results.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
            
    print("\nMetrics:")
    for m in all_metrics:
        print(f"\n{m['checkpoint']}:")
        for k, v in m.items():
            if k != "checkpoint":
                print(f"  {k}: {v:.4f}")
    
if __name__ == "__main__":
    main()
