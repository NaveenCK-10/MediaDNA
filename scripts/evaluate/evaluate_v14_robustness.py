"""
V14 Robustness Evaluation Script.

Generates corrupted versions of test_locked_v14.csv videos and evaluates 
the V14 checkpoint against them.
"""
import os
import sys
import csv
import json
import random
import subprocess
import shutil

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService
from backend.modules import analyze_visual_signals, extract_metadata, calculate_fusion

FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"

def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    random.seed(42)
    
    v14_ckpt = r"checkpoints\v14_fullscale\models\best_audio_model.pth"
    v8_ckpt = r"checkpoints\v8\baseline\models\best_audio_model.pth"
    checkpoint_path = v14_ckpt if os.path.exists(v14_ckpt) else v8_ckpt
    ckpt_name = "V14" if os.path.exists(v14_ckpt) else "V8"
    
    test_csv = "data/test_locked_v14.csv"
    if not os.path.exists(test_csv):
        test_csv = "data/test_locked_v8.csv"
    
    print(f"Checkpoint: {checkpoint_path} ({ckpt_name})")
    print(f"Test Set: {test_csv}")
    
    # Read test records
    records = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    # Sample 10 per category for corruption
    categories = ['RealVideo-RealAudio', 'FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio']
    sampled = []
    for c in categories:
        c_records = [r for r in records if r['type'] == c]
        if len(c_records) > 10:
            sampled.extend(random.sample(c_records, 10))
        else:
            sampled.extend(c_records)
    
    print(f"Sampled {len(sampled)} pristine videos for corruption.")
    
    # Generate corrupted files
    output_dir = "data/robustness_suite_v14"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    corruptions = [
        ("clean", []),
        ("video_crf40", ["-vcodec", "libx264", "-crf", "40", "-acodec", "copy"]),
        ("video_360p", ["-vf", "scale=-1:360", "-acodec", "copy"]),
        ("video_dark", ["-vf", "eq=brightness=-0.3", "-acodec", "copy"]),
        ("audio_aac_32k", ["-vcodec", "copy", "-acodec", "aac", "-b:a", "32k"]),
        ("audio_noise", ["-filter_complex", "aevalsrc=exprs=random(0):d=10[noise];[0:a][noise]amix=inputs=2:duration=first:dropout_transition=0", "-vcodec", "copy"])
    ]
    
    corruption_records = []
    for idx, row in enumerate(sampled):
        in_path = row['video_path']
        base_name = os.path.basename(in_path)
        name, ext = os.path.splitext(base_name)
        
        for corr_name, corr_args in corruptions:
            out_name = f"{name}_{corr_name}{ext}"
            out_path = os.path.join(output_dir, out_name)
            
            if corr_name == "clean":
                shutil.copy2(in_path, out_path)
                success = True
            else:
                cmd = [FFMPEG_PATH, "-y", "-i", in_path] + corr_args + [out_path]
                success = run_ffmpeg(cmd)
                
            if success:
                corruption_records.append({
                    "video_path": out_path,
                    "label": row['label'],
                    "type": row['type'],
                    "corruption": corr_name
                })
                
        if (idx+1) % 10 == 0:
            print(f"Generated corruptions for {idx+1}/{len(sampled)} videos...")
    
    print(f"Total corruption test cases: {len(corruption_records)}")
    
    # Now evaluate
    service = OpenAVFFService(checkpoint_path=checkpoint_path)
    
    results = []
    for idx, row in enumerate(corruption_records):
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
                "fusion_prob": fusion["mediadna_fake_prob"]
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
                "fusion_prob": -1
            })
            
        if (idx+1) % 20 == 0:
            print(f"Evaluated {idx+1}/{len(corruption_records)}")
            
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/v14_robustness_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "category", "corruption", "true_label", "prediction", "success",
            "openavff_prob", "visual_signal", "fusion_prob"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    import pandas as pd
    df = pd.DataFrame(results)
    df['success'] = df['success'].astype(bool)
    
    print("\n" + "="*60)
    print(f"V14 ROBUSTNESS RESULTS (Checkpoint: {ckpt_name})")
    print("="*60)
    
    grouped = df.groupby('corruption')['success'].agg(['mean', 'count'])
    for idx_row, row in grouped.iterrows():
        print(f"{idx_row} (N={int(row['count'])}): {row['mean']*100:.2f}% Accuracy")
    
    print("\n--- CATEGORY x CORRUPTION ---")
    grouped_cat = df.groupby(['corruption', 'category'])['success'].agg(['mean', 'count'])
    print(grouped_cat.to_string())
    
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()
