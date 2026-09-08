import cv2
import numpy as np
import os
import json

# Load thresholds from reports folder
THRESHOLD_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "visual_thresholds.json")
try:
    with open(THRESHOLD_FILE, "r") as f:
        thresholds = json.load(f)
        BLUR_CV_THRESHOLD = thresholds.get("blur_cv_threshold", 0.25)
        FRAME_MAE_STD_THRESHOLD = thresholds.get("frame_mae_std_threshold", 5.23)
except Exception:
    BLUR_CV_THRESHOLD = 0.25
    FRAME_MAE_STD_THRESHOLD = 5.23

def analyze_visual_signals(video_path: str, max_frames: int = 300) -> dict:
    """
    Calculates transparent visual signals from the video.
    Returns calculated values for UI display and an overall visual_anomaly_score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video {video_path}")
        
    blur_scores = []
    frame_diffs = []
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Video is empty")
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    blur_scores.append(cv2.Laplacian(prev_gray, cv2.CV_64F).var())
    
    count = 1
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur variance (Laplacian)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(blur)
        
        # Temporal difference (MAE between consecutive frames)
        diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
        frame_diffs.append(diff)
        
        prev_gray = gray
        count += 1
        
    cap.release()
    
    # Calculate final metrics
    if len(blur_scores) < 2:
        return {
            "blur_variance_mean": 0.0,
            "blur_coefficient_of_variation": 0.0,
            "frame_mae_mean": 0.0,
            "frame_mae_std": 0.0,
            "visual_anomaly_score": 0.0,
            "frames_analyzed": count,
            "thresholds_used": {"blur_cv": BLUR_CV_THRESHOLD, "frame_mae_std": FRAME_MAE_STD_THRESHOLD}
        }
        
    blur_mean = float(np.mean(blur_scores))
    blur_std = float(np.std(blur_scores))
    blur_cv = blur_std / blur_mean if blur_mean > 0 else 0.0
    
    diff_std = float(np.std(frame_diffs))
    diff_mean = float(np.mean(frame_diffs))
    
    # Normalization: Map to 0-1 based on calibrated thresholds.
    # If the value is below 50% of the threshold, anomaly is 0 (safe).
    # If it reaches the 95th percentile threshold, anomaly is 1 (highly suspicious).
    blur_anomaly = min(1.0, max(0.0, (blur_cv - (BLUR_CV_THRESHOLD * 0.5)) / (BLUR_CV_THRESHOLD * 0.5))) if BLUR_CV_THRESHOLD > 0 else 0.0
    temporal_anomaly = min(1.0, max(0.0, (diff_std - (FRAME_MAE_STD_THRESHOLD * 0.5)) / (FRAME_MAE_STD_THRESHOLD * 0.5))) if FRAME_MAE_STD_THRESHOLD > 0 else 0.0
    
    visual_anomaly_score = (blur_anomaly * 0.5) + (temporal_anomaly * 0.5)
    
    return {
        "blur_variance_mean": blur_mean,
        "blur_coefficient_of_variation": float(blur_cv),
        "frame_mae_mean": diff_mean,
        "frame_mae_std": diff_std,
        "visual_anomaly_score": float(visual_anomaly_score),
        "frames_analyzed": count,
        "thresholds_used": {"blur_cv": BLUR_CV_THRESHOLD, "frame_mae_std": FRAME_MAE_STD_THRESHOLD}
    }
