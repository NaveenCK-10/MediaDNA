import os
import json
from backend.modules import analyze_visual_signals, extract_metadata
import pandas as pd
import numpy as np

# Load validation set
val_df = pd.read_csv('data/val_v14.csv')
# Get 50 real videos
real_vids = val_df[val_df['label'] == 0]['video_path'].head(50).tolist()

vis_scores = []
blur_cvs = []
mae_means = []

for vid in real_vids:
    if os.path.exists(vid):
        try:
            vis = analyze_visual_signals(vid)
            vis_scores.append(vis['visual_anomaly_score'])
            blur_cvs.append(vis['blur_coefficient_of_variation'])
            mae_means.append(vis['frame_mae_mean'])
        except Exception as e:
            pass

target = analyze_visual_signals('src/naveen.mp4')

print(f"Naveen Visual Score: {target['visual_anomaly_score']:.4f}")
print(f"Naveen Blur CV: {target['blur_coefficient_of_variation']:.4f}")
print(f"Naveen MAE Mean: {target['frame_mae_mean']:.4f}")

print(f"\nReal Validation Set (N={len(vis_scores)}):")
print(f"Mean Visual Score: {np.mean(vis_scores):.4f}")
print(f"Mean Blur CV: {np.mean(blur_cvs):.4f}")
print(f"Mean MAE Mean: {np.mean(mae_means):.4f}")
