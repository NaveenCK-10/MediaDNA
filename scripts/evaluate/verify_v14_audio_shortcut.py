import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService

def evaluate_audio_shortcut():
    checkpoint_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\checkpoints\v14_fullscale\models\best_audio_model.pth"
    csv_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\test_locked_v14.csv"
    
    # Use V8 if V14 is still training for testing the script structure
    if not os.path.exists(checkpoint_path):
        checkpoint_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\checkpoints\v8\baseline\models\best_audio_model.pth"
        print("V14 checkpoint not found, falling back to V8 for audio shortcut test.")
        
    if not os.path.exists(csv_path):
        print(f"ERROR: Test CSV not found at {csv_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    service = OpenAVFFService(checkpoint_path=checkpoint_path)
    
    df = pd.read_csv(csv_path)
    
    # We will subclass or monkeypatch OpenAVFFService's analyze_video 
    # to zero out the video tensor before inference.
    
    original_analyze = service.analyze_video
    
    def analyze_audio_only(video_path: str):
        # Override the analyze_video temporarily
        total_start = __import__('time').time()
        video_filename = os.path.basename(video_path)
        fbank = service._extract_audio_fbank(video_path)
        frames, num_frames = service._extract_video_frames(video_path)
        
        a_input = fbank.unsqueeze(0).to(service.device)
        
        # ZERO OUT VIDEO TENSOR
        v_input = torch.zeros_like(frames.unsqueeze(0)).to(service.device)
        
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = service.model(a_input, v_input)
                
        probabilities = torch.sigmoid(output).cpu().float().numpy()[0]
        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])
        return 1 if fake_prob >= 0.5 else 0, fake_prob

    # Test Audio Shortcut
    print(f"\n--- EVALUATING AUDIO SHORTCUT ON {len(df)} VIDEOS ---")
    y_true = []
    y_pred = []
    y_prob = []
    
    for idx, row in df.iterrows():
        video_path = os.path.join(r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF", row['video_path'])
        label = int(row['label'])
        
        try:
            pred, prob = analyze_audio_only(video_path)
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)
            
            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(df)} videos...")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')
        
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- AUDIO-ONLY METRICS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save the report
    os.makedirs("reports", exist_ok=True)
    with open("reports/v14_audio_shortcut_report.txt", "w") as f:
        f.write("--- AUDIO-ONLY INFERENCE METRICS ---\n")
        f.write("Method: Video tensor explicitly zeroed out.\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

if __name__ == "__main__":
    evaluate_audio_shortcut()
