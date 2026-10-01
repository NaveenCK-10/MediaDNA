import os, sys, csv, json, pickle
import torch
import torch.nn as nn
import numpy as np
import subprocess, torchaudio, soundfile as sf
import torchvision.transforms as T
from decord import VideoReader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"

def extract_audio(video_path):
    temp_wav = f"temp_cal_{os.getpid()}.wav"
    try:
        cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        if not os.path.exists(temp_wav): return torch.zeros(1024, 128)
        wav, sr = sf.read(temp_wav)
        if len(wav.shape) > 1: wav = wav.mean(axis=1)
        audio_tensor = torch.FloatTensor(wav)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128)(audio_tensor)
        mel = (mel + 1e-6).log()
        mel = (mel - (-5.081)) / 4.4849
        if mel.shape[1] < 1024:
            mel = nn.functional.pad(mel, (0, 1024 - mel.shape[1]))
        else:
            mel = mel[:, :1024]
        if os.path.exists(temp_wav): os.remove(temp_wav)
        return mel.transpose(0, 1)
    except:
        if os.path.exists(temp_wav): os.remove(temp_wav)
        return torch.zeros(1024, 128)

def main():
    print("=== FINAL MODEL CALIBRATION ===")
    # Load the best model (assume AVFF for now, but we can change if retrained is better)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    # if retrained is better, we would load V22_4F_MULTIMODAL_CHECKPOINT.pth
    
    model = VideoCAVMAEFT()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.to(DEVICE)
    model.eval()
    
    transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    with open(os.path.join(PROJECT_ROOT, "data", "v22_2_calibration.csv"), 'r') as f:
        rows = list(csv.DictReader(f))
        
    logits = []
    labels = []
    
    print(f"Processing {len(rows)} calibration samples...")
    for i, r in enumerate(rows):
        fname = r["path"]
        dir_col = r.get("", "") or [v for k,v in r.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        full_path = os.path.join(DATASET_DIR, dir_col.replace("FakeAVCeleb/", ""), fname)
        label = 0 if r["type"] == "RealVideo-RealAudio" else 1
        
        a_t = extract_audio(full_path).unsqueeze(0).unsqueeze(1).to(DEVICE)
        
        try:
            vr = VideoReader(full_path, width=224, height=224)
            frame_idx = np.linspace(0, len(vr) - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_t = transform(v_t).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
        except:
            continue
            
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(a_t, v_t)
                prob = torch.sigmoid(out).item()
                logit = np.log(prob / (1 - prob + 1e-12) + 1e-12)
                logits.append(logit)
                labels.append(label)
                
        if (i+1) % 100 == 0: print(f"Processed {i+1}/{len(rows)}")
        
    X = np.array(logits).reshape(-1, 1)
    y = np.array(labels)
    
    calibrator = LogisticRegression(solver='lbfgs')
    calibrator.fit(X, y)
    
    cal_probs = calibrator.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, cal_probs)
    
    n_real = sum(1 for l in y if l == 0)
    n_fake = sum(1 for l in y if l == 1)
    
    print(f"Fit Platt Calibrator on {len(y)} CAL samples ({n_real} real, {n_fake} fake)")
    print(f"Brier Score: {brier:.4f}")
    
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "FINAL_CALIBRATOR.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
        
    report = f"""# MediaDNA Final Calibration Report

## Calibration Dataset
- **Split**: `v22_2_calibration.csv` ONLY
- **Sample Count**: {len(y)} successfully extracted
- **Class Balance**: {n_real} Real, {n_fake} Synthetic

## Calibration Method
Platt Scaling (Logistic Regression) fit strictly on the held-out CAL split logits.

## Performance
- **Brier Score Loss**: {brier:.4f}
"""
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "MEDIADNA_FINAL_CALIBRATION_REPORT.md"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
