import os
import sys
import glob
import csv
import json
import random
import numpy as np
import cv2

def extract_visual_raw(video_path, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
        
    blur_scores = []
    frame_diffs = []
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None, None
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    blur_scores.append(cv2.Laplacian(prev_gray, cv2.CV_64F).var())
    
    count = 1
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(blur)
        
        diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
        frame_diffs.append(diff)
        
        prev_gray = gray
        count += 1
        
    cap.release()
    
    if len(blur_scores) < 2:
        return 0.0, 0.0
        
    blur_mean = np.mean(blur_scores)
    blur_std = np.std(blur_scores)
    blur_cv = blur_std / blur_mean if blur_mean > 0 else 0
    
    diff_std = np.std(frame_diffs)
    
    return float(blur_cv), float(diff_std)

def main():
    base_dir = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    train_csv = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\train_medium.csv"
    
    train_paths = []
    try:
        with open(train_csv, "r") as f:
            for row in csv.reader(f):
                if row:
                    train_paths.append(row[0].strip())
    except Exception as e:
        print(f"Failed to read train CSV: {e}")
        return
        
    # Separate Reals and Fakes
    reals = [p for p in train_paths if "RealVideo" in p]
    fakes = [p for p in train_paths if "FakeVideo" in p]
    
    random.seed(42)
    sample_reals = random.sample(reals, min(50, len(reals)))
    sample_fakes = random.sample(fakes, min(50, len(fakes)))
    
    real_blurs = []
    real_diffs = []
    fake_blurs = []
    fake_diffs = []
    
    print(f"Calibrating on {len(sample_reals)} Real and {len(sample_fakes)} Fake training videos...")
    
    for path in sample_reals:
        full = os.path.join(base_dir, path)
        if os.path.exists(full):
            b, d = extract_visual_raw(full)
            if b is not None:
                real_blurs.append(b)
                real_diffs.append(d)
                
    for path in sample_fakes:
        full = os.path.join(base_dir, path)
        if os.path.exists(full):
            b, d = extract_visual_raw(full)
            if b is not None:
                fake_blurs.append(b)
                fake_diffs.append(d)
                
    # We want anomalous thresholds. 95th percentile of REAL videos.
    blur_threshold = float(np.percentile(real_blurs, 95)) if real_blurs else 0.5
    diff_threshold = float(np.percentile(real_diffs, 95)) if real_diffs else 10.0
    
    # Let's ensure a minimum threshold so it's not too sensitive
    blur_threshold = max(blur_threshold, 0.2)
    diff_threshold = max(diff_threshold, 3.0)
    
    result = {
        "blur_cv_threshold": blur_threshold,
        "frame_mae_std_threshold": diff_threshold,
        "description": "Calculated as max(minimum_baseline, 95th percentile of real training videos) to limit false positive anomalies."
    }
    
    os.makedirs("reports", exist_ok=True)
    out_path = r"reports\visual_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
        
    print(f"Calibration complete. Thresholds saved to {out_path}:")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
