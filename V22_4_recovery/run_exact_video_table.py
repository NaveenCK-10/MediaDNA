import os
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import subprocess
import torchaudio
import soundfile as sf
import torchvision.transforms as T
from decord import VideoReader
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AVFF_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
CALIBRATOR_PATH = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_PLATT_CALIBRATOR.pkl")
POLICY_PATH = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_THRESHOLD_POLICY.json")

def process_video(video_path):
    if not os.path.exists(video_path):
        return None, None
    try:
        vr = VideoReader(video_path, width=224, height=224)
        num_frames = len(vr)
        frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
        v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        v_tensor = transform(v_tensor).permute(1, 0, 2, 3).unsqueeze(0)
        
        temp_wav = f"temp_demo_tb_{os.getpid()}.wav"
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
        return None, None

def find_video(name):
    if os.path.exists(name): return name
    for root, dirs, files in os.walk(DATASET_DIR):
        if name in files:
            return os.path.join(root, name)
    return None

def main():
    model = VideoCAVMAEFT()
    model = nn.DataParallel(model)
    ckpt = torch.load(AVFF_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()

    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)
    
    with open(POLICY_PATH, "r") as f:
        policy = json.load(f)

    videos = [
        ("Create_a_photorealistic_AI_gen.mp4", r"C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4", "SYNTHETIC"),
        ("00143_clean.mp4", None, "AUTHENTIC"),
        ("00160_id01098_wavtolip_clean.mp4", None, "SYNTHETIC")
    ]

    out_csv = os.path.join(PROJECT_ROOT, "V22_4_recovery", "MEDIADNA_V22_4_FINAL_EXACT_VIDEO_TABLE.csv")
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["VIDEO", "LABEL IF KNOWN", "VISUAL", "AUDIO", "FUSION", "CALIBRATED", "DECISION"])
        
        for name, path, label in videos:
            if path is None:
                path = find_video(name)
            
            if path is None:
                writer.writerow([name, label, "N/A", "N/A", "N/A", "N/A", "FILE_NOT_FOUND"])
                continue
                
            a_t, v_t = process_video(path)
            if a_t is None:
                writer.writerow([name, label, "N/A", "N/A", "N/A", "N/A", "PROCESSING_ERROR"])
                continue
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    out = model(a_t.to(DEVICE), v_t.to(DEVICE))
                prob = float(torch.sigmoid(out).cpu().float().numpy()[0][0])
                
            logit = np.log(prob / (1 - prob + 1e-12) + 1e-12)
            cal_prob = float(calibrator.predict_proba(np.array([[logit]]))[0][1])
            
            if cal_prob >= policy["synthetic_lower"]: dec = "SYNTHETIC"
            elif cal_prob < policy["authentic_upper"]: dec = "AUTHENTIC"
            else: dec = "UNCERTAIN"
            
            # Since AVFF doesn't separate visual/audio natively without complex probing, we put N/A or report fusion.
            # We will report fusion score for all 3 as per standard AVFF fallback
            writer.writerow([name, label, "N/A", "N/A", f"{prob:.4f}", f"{cal_prob:.4f}", dec])

if __name__ == "__main__":
    main()
