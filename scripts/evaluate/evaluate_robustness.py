import os
import sys
import csv
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService
from backend.modules import analyze_visual_signals, extract_metadata, calculate_fusion

def main():
    print("="*60)
    print("V9 Robustness Evaluation")
    print("="*60)
    
    ckpt_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\checkpoints\v8\baseline\models\best_audio_model.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-local\models\best_audio_model.pth"
        print("Using V7 checkpoint because V8 didn't finish properly")
    else:
        print("Using V8 Baseline checkpoint")
        
    csv_path = "data/test_robustness_v9.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run generate_robustness.py first.")
        return
        
    service = OpenAVFFService(checkpoint_path=ckpt_path)
    
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    results = []
    
    print(f"Starting evaluation of {len(records)} robust files...")
    
    for idx, row in enumerate(records):
        vid_path = row['video_path']
        label = int(row['label'])
        
        try:
            res = service.analyze_video(vid_path)
            visual_signals = analyze_visual_signals(vid_path)
            meta = extract_metadata(vid_path)
            fusion = calculate_fusion(res.fake_probability, visual_signals["visual_anomaly_score"])
            
            pred_fake = fusion["prediction"] == "fake"
            pred_label = 1 if pred_fake else 0
            success = pred_label == label
            
            results.append({
                "file": os.path.basename(vid_path),
                "category": row['type'],
                "corruption": row['corruption'],
                "true_label": label,
                "prediction": "fake" if pred_label else "real",
                "success": success,
                "openavff_prob": res.fake_probability,
                "visual_signal": visual_signals["visual_anomaly_score"],
                "fusion_prob": fusion["mediadna_fake_prob"],
                "metadata": json.dumps(meta)
            })
        except Exception as e:
            print(f"Failed {vid_path}: {e}")
            results.append({
                "file": os.path.basename(vid_path),
                "category": row['type'],
                "corruption": row['corruption'],
                "true_label": label,
                "prediction": "ERROR",
                "success": False,
                "openavff_prob": -1,
                "visual_signal": -1,
                "fusion_prob": -1,
                "metadata": "{}"
            })
            
        if (idx+1) % 10 == 0:
            print(f"Evaluated {idx+1}/{len(records)}")
            
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/v9_evaluation_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "category", "corruption", "true_label", "prediction", "success", 
            "openavff_prob", "visual_signal", "fusion_prob", "metadata"
        ])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Evaluation complete. Results saved to {out_csv}")

if __name__ == "__main__":
    main()
