import os
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, average_precision_score)
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
LOCKED_CSV = os.path.join(PROJECT_ROOT, "data", "v22_2_test_locked.csv")
CALIBRATOR_PATH = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_PLATT_CALIBRATOR.pkl")
POLICY_PATH = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_THRESHOLD_POLICY.json")


def process_video(video_path):
    try:
        vr = VideoReader(video_path, width=224, height=224)
        num_frames = len(vr)
        frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
        v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        v_tensor = transform(v_tensor).permute(1, 0, 2, 3).unsqueeze(0)
        
        temp_wav = f"temp_lk_{os.getpid()}.wav"
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


def main():
    print("Loading AVFF model for Locked Test...", flush=True)
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

    with open(LOCKED_CSV, 'r') as f:
        rows = list(csv.DictReader(f))
    
    # We will sample 1/4th of the locked test to speed it up if time is critical
    # Wait, the prompt says "evaluate LOCKED TEST exactly once". Better to do the whole thing.
    
    preds, labels, cal_probs = [], [], []
    errors = 0
    
    print(f"Total locked samples: {len(rows)}", flush=True)
    for i, row in enumerate(rows):
        fname = row["path"]
        dir_col = row.get("", "") or [v for k,v in row.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        rel_dir = dir_col.replace("FakeAVCeleb/", "")
        full_path = os.path.join(DATASET_DIR, rel_dir, fname)
        label = 0 if row["type"] == "RealVideo-RealAudio" else 1
        
        a_t, v_t = process_video(full_path)
        if a_t is None:
            errors += 1
            continue
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(a_t.to(DEVICE), v_t.to(DEVICE))
            prob = float(torch.sigmoid(out).cpu().float().numpy()[0][0])
            
        logit = np.log(prob / (1 - prob + 1e-12) + 1e-12)
        cal_prob = float(calibrator.predict_proba(np.array([[logit]]))[0][1])
        
        preds.append(prob)
        cal_probs.append(cal_prob)
        labels.append(label)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(rows)}", flush=True)

    labels = np.array(labels)
    preds = np.array(preds)
    cal_probs = np.array(cal_probs)
    
    thresh = policy["synthetic_lower"]
    p_bin = (cal_probs >= thresh).astype(int)
    
    metrics = {
        "ROC_AUC": float(roc_auc_score(labels, preds)),
        "PR_AUC": float(average_precision_score(labels, preds)),
        "Balanced_Accuracy": float(balanced_accuracy_score(labels, p_bin)),
        "MCC": float(matthews_corrcoef(labels, p_bin)),
        "Precision": float(precision_score(labels, p_bin, zero_division=0)),
        "Recall": float(recall_score(labels, p_bin, zero_division=0)),
        "F1": float(f1_score(labels, p_bin, zero_division=0))
    }
    
    tn, fp, fn, tp = confusion_matrix(labels, p_bin).ravel()
    metrics["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["Confusion_Matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    
    res = {
        "metrics": metrics,
        "errors": errors,
        "n_samples": len(preds)
    }
    
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_LOCKED_TEST_METRICS.json"), "w") as f:
        json.dump(res, f, indent=2)
        
    print(json.dumps(res, indent=2), flush=True)

if __name__ == "__main__":
    main()
