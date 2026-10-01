"""
V22.4 Fast Calibration & Demo Check
Fits Platt scaling using the existing V22.2 DEV predictions to save time,
then runs the final demo test on the user video.
"""
import os, sys, csv, json, pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
import subprocess
import torchaudio
import soundfile as sf
import torchvision.transforms as T
from decord import VideoReader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AVFF_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")

def process_video(video_path):
    try:
        vr = VideoReader(video_path, width=224, height=224)
        num_frames = len(vr)
        frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
        v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        v_tensor = transform(v_tensor).permute(1, 0, 2, 3).unsqueeze(0)
        
        temp_wav = f"temp_fast_demo_{os.getpid()}.wav"
        cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        
        if not os.path.exists(temp_wav):
            a_tensor = torch.zeros(1, 1024, 128)
        else:
            wav, sr = sf.read(temp_wav)
            if len(wav.shape) > 1: wav = wav.mean(axis=1)
            audio_tensor = torch.FloatTensor(wav)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128
            )(audio_tensor)
            mel_spec = (mel_spec + 1e-6).log()
            mel_spec = (mel_spec - (-5.081)) / 4.4849
            target_len = 1024
            if mel_spec.shape[1] < target_len:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, target_len - mel_spec.shape[1]))
            else:
                mel_spec = mel_spec[:, :target_len]
            mel_spec = mel_spec.transpose(0, 1)
            a_tensor = mel_spec.unsqueeze(0)
            os.remove(temp_wav)
        return a_tensor, v_tensor
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None, None

def main():
    print("Fitting Fast Platt Calibrator on DEV predictions...")
    pred_csv = os.path.join(PROJECT_ROOT, "V22_2_BASELINE_PREDICTIONS.csv")
    
    preds, labels = [], []
    with open(pred_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(float(row["prob"]))
            labels.append(int(row["label"]))
            
    preds = np.array(preds)
    labels = np.array(labels)
    logits = np.log(preds / (1 - preds + 1e-12) + 1e-12).reshape(-1, 1)
    
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(logits, labels)
    
    cal_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_PLATT_CALIBRATOR.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
        
    policy = {
        "authentic_upper": 0.35,
        "synthetic_lower": 0.65,
        "calibrated_on": "V22_2_DEV"
    }
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_THRESHOLD_POLICY.json"), "w") as f:
        json.dump(policy, f, indent=2)
        
    print("Calibrator fit and saved. Now testing demo video...")
    
    model = VideoCAVMAEFT()
    model = nn.DataParallel(model)
    ckpt = torch.load(AVFF_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    
    demo_vid = r"C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4"
    a_t, v_t = process_video(demo_vid)
    if a_t is not None:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(a_t.to(DEVICE), v_t.to(DEVICE))
            raw_prob = float(torch.sigmoid(out).cpu().numpy()[0][0])
            
        logit = np.log(raw_prob / (1 - raw_prob + 1e-12) + 1e-12)
        cal_prob = float(calibrator.predict_proba(np.array([[logit]]))[0][1])
        
        print("\n" + "="*50)
        print("DEMO SANITY CHECK RESULT")
        print("="*50)
        print(f"Video:      {os.path.basename(demo_vid)}")
        print(f"Raw Prob:   {raw_prob:.6f}")
        print(f"Calibrated: {cal_prob:.6f}")
        
        if cal_prob >= policy["synthetic_lower"]: dec = "SYNTHETIC"
        elif cal_prob < policy["authentic_upper"]: dec = "AUTHENTIC"
        else: dec = "UNCERTAIN"
        
        print(f"Decision:   {dec}")
        print("="*50)

if __name__ == "__main__":
    main()
