import os
import sys
import time
import csv
import json
import hashlib
import torch
import torchaudio
import numpy as np
import soundfile as sf
import torchvision.transforms as T
from decord import VideoReader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

# PREPROCESSING CONFIG
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AUDIO_SAMPLE_RATE = 16000
TARGET_LENGTH = 1024
NUM_MEL_BINS = 128
IM_RES = 224
NUM_FRAMES = 16
DATASET_MEAN = -5.081
DATASET_STD = 4.4849
THRESHOLD = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class MinimalDeterministicBenchmark:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        print(f"Loading checkpoint: {checkpoint_path}")
        self.model = VideoCAVMAEFT()
        self.model = torch.nn.DataParallel(self.model)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(DEVICE)
        self.model.eval()

    def _extract_audio_fbank(self, video_path):
        import io
        cmd = [FFMPEG_PATH, "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "wav", "pipe:1"]
        result = subprocess.run(cmd, check=True, capture_output=True)
        waveform, sr = sf.read(io.BytesIO(result.stdout))
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1, 2), size=(TARGET_LENGTH,), mode="linear", align_corners=False).transpose(1, 2).squeeze(0)
        return (fbank - DATASET_MEAN) / DATASET_STD

    def _extract_video_frames(self, video_path):
        vr = VideoReader(video_path)
        frame_indices = np.linspace(0, len(vr) - 1, NUM_FRAMES).astype(int)
        frames = [vr[i].asnumpy() for i in frame_indices]
        tf = T.Compose([T.ToPILImage(), T.Resize((IM_RES, IM_RES)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        frames = torch.stack([tf(f) for f in frames]).permute(1, 0, 2, 3)
        return frames

    def infer(self, video_path):
        try:
            fbank = self._extract_audio_fbank(video_path)
            frames = self._extract_video_frames(video_path)
            
            a_input = fbank.unsqueeze(0).to(DEVICE)
            v_input = frames.unsqueeze(0).to(DEVICE)
            
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    output = self.model(a_input, v_input)
            
            base_probs = torch.sigmoid(output).cpu().float().numpy()[0]
            return float(base_probs[0])
        except Exception as e:
            print(f"Error inferring {video_path}: {e}")
            return None

def write_preprocessing_config():
    config = {
        "audio": {
            "sample_rate": AUDIO_SAMPLE_RATE,
            "mel_bins": NUM_MEL_BINS,
            "target_length": TARGET_LENGTH,
            "mean": DATASET_MEAN,
            "std": DATASET_STD,
            "hop_length": 10,
            "normalization": "zero_mean_wave_and_zscore_fbank"
        },
        "video": {
            "frames": NUM_FRAMES,
            "resolution": IM_RES,
            "normalization": "imagenet",
            "crop": "none_resized_direct",
            "stride": "uniform_linspace"
        },
        "fusion": "concatenation_before_mlp"
    }
    with open("V21_4_PREPROCESSING_CONFIG.json", "w") as f:
        json.dump(config, f, indent=4)

def run_benchmark(run_id=""):
    print(f"Starting Benchmark Run {run_id}...")
    
    # Checkpoint validation
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    ckpt_hash = compute_sha256(ckpt_path)
    print(f"Checkpoint SHA256: {ckpt_hash}")

    write_preprocessing_config()
    
    # Load manifest
    manifest_path = "V21_4_LOCKED_TEST_MANIFEST.csv"
    manifest = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest.append(row)
            
    print(f"Loaded {len(manifest)} manifest samples.")
    
    benchmark = MinimalDeterministicBenchmark(ckpt_path)
    
    predictions = []
    for item in manifest:
        score = benchmark.infer(item["path"])
        if score is None:
            continue
            
        pred = 1 if score >= THRESHOLD else 0
        predictions.append({
            "sample_id": item["sample_id"],
            "sha256": item["sha256"],
            "ground_truth": item["ground_truth"],
            "score": score,
            "prediction": pred,
            "identity": item["identity"],
            "source": item["source_video"],
            "video_category": item["video_manipulation"],
            "audio_category": item["audio_manipulation"]
        })
        
    out_file = f"V21_4_PREDICTIONS{run_id}.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
        
    print(f"Predictions saved to {out_file}")
    
    return predictions, ckpt_hash

def compute_metrics(predictions):
    y_true = [int(p["ground_truth"]) for p in predictions]
    y_pred = [int(p["prediction"]) for p in predictions]
    y_score = [float(p["score"]) for p in predictions]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float('nan')
    except:
        auc = float('nan')
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    balanced_acc = (rec + spec) / 2
    
    return {
        "Accuracy": acc,
        "Balanced_Accuracy": balanced_acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "MCC": mcc,
        "NPV": npv,
        "ROC-AUC": auc,
        "FPR": fpr,
        "FNR": fnr,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

if __name__ == "__main__":
    preds1, ckpt_hash = run_benchmark("_run1")
    preds2, _ = run_benchmark("_run2")
    
    # Verify Determinism
    hash1 = compute_sha256("V21_4_PREDICTIONS_run1.csv")
    hash2 = compute_sha256("V21_4_PREDICTIONS_run2.csv")
    is_deterministic = hash1 == hash2
    
    metrics = compute_metrics(preds1)
    
    res = {
        "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
        "checkpoint_hash": ckpt_hash,
        "is_deterministic": is_deterministic,
        "manifest_size": len(preds1),
        "metrics": metrics
    }
    
    with open("V21_4_BASELINE_RESULTS.json", "w") as f:
        json.dump(res, f, indent=4)
        
    print("V21.4 Benchmark Completed.")
