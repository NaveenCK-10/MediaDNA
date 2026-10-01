import os, sys, csv, time, json, pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
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
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            self.rows = list(csv.DictReader(f))
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.labels = []
        for r in self.rows:
            self.labels.append(0 if r["type"] == "RealVideo-RealAudio" else 1)

    def __len__(self): return len(self.rows)

    def _extract_mel(self, video_path):
        temp_wav = f"temp_aud_{os.getpid()}_{id(self)}_{time.time()}.wav"
        try:
            cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            if not os.path.exists(temp_wav): return torch.zeros(1024, 128)
            wav, sr = sf.read(temp_wav)
            if len(wav.shape) > 1: wav = wav.mean(axis=1)
            audio_tensor = torch.FloatTensor(wav)
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128)(audio_tensor)
            mel_spec = (mel_spec + 1e-6).log()
            mel_spec = (mel_spec - (-5.081)) / 4.4849
            
            target_len = 1024
            if mel_spec.shape[1] < target_len:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, target_len - mel_spec.shape[1]))
            else:
                mel_spec = mel_spec[:, :target_len]
            mel_spec = mel_spec.transpose(0, 1)
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return mel_spec
        except:
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return torch.zeros(1024, 128)

    def __getitem__(self, idx):
        r = self.rows[idx]
        fname = r["path"]
        dir_col = r.get("", "") or [v for k,v in r.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        full_path = os.path.join(DATASET_DIR, dir_col.replace("FakeAVCeleb/", ""), fname)
        label = 0 if r["type"] == "RealVideo-RealAudio" else 1
        
        try:
            vr = VideoReader(full_path, width=224, height=224)
            frame_idx = np.linspace(0, len(vr) - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_t = self.transform(v_t).permute(1, 0, 2, 3)
        except:
            v_t = torch.zeros(3, 16, 224, 224)
            
        a_t = self._extract_mel(full_path)
        return a_t, v_t, torch.tensor([label], dtype=torch.float32), full_path

def evaluate_dataset(model, dataloader, name=""):
    model.eval()
    preds, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (a, v, l, _) in enumerate(dataloader):
            if i % 100 == 0: print(f"  {name} progress: {i}/{len(dataloader)} batches")
            a, v = a.to(DEVICE), v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
    
    preds = np.array(preds)
    labels = np.array(labels)
    dur = time.time() - t0
    
    return preds, labels, dur

def verify_metrics(preds, labels):
    p_bin = (preds >= 0.5).astype(int)
    auc = roc_auc_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    bacc = balanced_accuracy_score(labels, p_bin)
    mcc = matthews_corrcoef(labels, p_bin)
    prec = precision_score(labels, p_bin, zero_division=0)
    rec = recall_score(labels, p_bin, zero_division=0)
    cm = confusion_matrix(labels, p_bin)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    f1 = f1_score(labels, p_bin, zero_division=0)
    
    print("\nVerification Results:")
    print(f"ROC-AUC:           {auc:.4f} (Expected: 0.7292)")
    print(f"PR-AUC:            {pr_auc:.4f} (Expected: 0.9877)")
    print(f"Balanced Accuracy: {bacc:.4f} (Expected: 0.7387)")
    print(f"MCC:               {mcc:.4f} (Expected: 0.1436)")
    print(f"Precision:         {prec:.4f} (Expected: 0.9949)")
    print(f"Recall:            {rec:.4f} (Expected: 0.6174)")
    print(f"Specificity:       {spec:.4f} (Expected: 0.8600)")
    print(f"Confusion Matrix:\n{cm}\n(Expected: [[43, 7], [847, 1367]])")
    
    # Assert reproducible
    assert abs(auc - 0.7292) < 0.005, "AUC mismatch!"
    assert abs(mcc - 0.1436) < 0.005, "MCC mismatch!"
    print("Reproducibility check PASSED.\n")

def run_calibration(cal_preds, cal_labels, dev_preds, dev_labels):
    print("Fitting Platt Scaling on CAL...")
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(cal_preds.reshape(-1, 1), cal_labels)
    
    # Apply to DEV
    dev_cal_preds = lr.predict_proba(dev_preds.reshape(-1, 1))[:, 1]
    
    print("\n--- DEV Calibrated Metrics ---")
    auc = roc_auc_score(dev_labels, dev_cal_preds)
    p_bin = (dev_cal_preds >= 0.5).astype(int)
    cm = confusion_matrix(dev_labels, p_bin)
    print(f"Calibrated ROC-AUC: {auc:.4f}")
    print(f"Calibrated BAcc:    {balanced_accuracy_score(dev_labels, p_bin):.4f}")
    print(f"Calibrated MCC:     {matthews_corrcoef(dev_labels, p_bin):.4f}")
    print(f"Calibrated CM:\n{cm}")
    
    return lr, dev_cal_preds

def evaluate_thresholds(preds, labels):
    print("\nEvaluating 3-state Thresholds on DEV...")
    
    # Grid search for policy that keeps False Positives < 2 (Specificity > 95%) and captures maximum highly confident fakes
    thresholds_low = np.linspace(0.1, 0.4, 30)
    thresholds_high = np.linspace(0.6, 0.95, 35)
    
    best_policy = None
    best_f1 = -1
    
    for t_low in thresholds_low:
        for t_high in thresholds_high:
            if t_low >= t_high: continue
            
            p_bin = np.zeros_like(preds)
            p_bin[preds > t_high] = 1
            p_bin[(preds <= t_high) & (preds >= t_low)] = 2 # Suspect
            
            # For metrics, only count confident Real (0) and confident Fake (1)
            confident_idx = (p_bin == 0) | (p_bin == 1)
            
            if np.sum(confident_idx) < len(preds) * 0.1: continue # require at least 10% coverage
            
            # Specificity (Confident Real / Total Actual Real among confident)
            y_conf = labels[confident_idx]
            p_conf = p_bin[confident_idx]
            cm = confusion_matrix(y_conf, p_conf)
            if cm.shape != (2,2): continue
            
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp + 1e-6)
            f1 = f1_score(y_conf, p_conf)
            
            # Policy rules
            if spec >= 0.95 and fp <= 2:
                if f1 > best_f1:
                    best_f1 = f1
                    best_policy = (t_low, t_high, spec, fp, tn, fn, tp, np.sum(confident_idx)/len(preds))

    if best_policy is None:
        print("Could not find a 95% specificity policy. Relaxing constraints...")
        best_policy = (0.3, 0.7, 0.0, 0, 0, 0, 0, 1.0)
    else:
        t_low, t_high, spec, fp, tn, fn, tp, cov = best_policy
        print(f"Selected Policy: Low={t_low:.4f}, High={t_high:.4f}")
        print(f"Coverage: {cov*100:.1f}%")
        print(f"Confident Real: {tn} (True Real) vs {fn} (False Real)")
        print(f"Confident Fake: {tp} (True Fake) vs {fp} (False Fake - FATAL)")
        print(f"Confident Specificity: {spec:.4f}")
    
    return t_low, t_high

def main():
    print("=" * 60)
    print("PHASE 6: CALIBRATION & THRESHOLDING (V22.4F)")
    print("=" * 60)
    
    # 1. Load Checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4F_MULTIMODAL_StageB_Ep2.pth")
    model = VideoCAVMAEFT()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.to(DEVICE)
    print("Loaded StageB_Ep2 checkpoint successfully.")
    
    # 2. Re-run DEV and verify
    dev_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    print("\nRunning full DEV Evaluation (2264 samples)...")
    dev_preds, dev_labels, dev_dur = evaluate_dataset(model, dev_loader, "DEV")
    verify_metrics(dev_preds, dev_labels)
    
    # 3. Generate CAL scores
    cal_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_calibration.csv"))
    cal_loader = DataLoader(cal_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    print("\nRunning full CAL Evaluation (2195 samples)...")
    cal_preds, cal_labels, cal_dur = evaluate_dataset(model, cal_loader, "CAL")
    
    # 4 & 5. Platt Scaling
    calibrator, dev_cal_preds = run_calibration(cal_preds, cal_labels, dev_preds, dev_labels)
    
    # Save Calibrator
    calibrator_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4F_PLATT_CALIBRATOR.pkl")
    with open(calibrator_path, "wb") as f: pickle.dump(calibrator, f)
    print(f"\nSaved calibrator to {calibrator_path}")
    
    # 6 & 7. 3-state thresholds
    t_low, t_high = evaluate_thresholds(dev_cal_preds, dev_labels)
    
    policy = {
        "model": "V22_4F_Multimodal",
        "checkpoint": "V22_4F_MULTIMODAL_StageB_Ep2.pth",
        "thresholds": {
            "low_real": t_low,
            "high_fake": t_high
        },
        "states": {
            "real": f"prob < {t_low:.4f}",
            "suspect": f"{t_low:.4f} <= prob <= {t_high:.4f}",
            "fake": f"prob > {t_high:.4f}"
        }
    }
    policy_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4F_THRESHOLD_POLICY.json")
    with open(policy_path, "w") as f: json.dump(policy, f, indent=4)
    print(f"Saved threshold policy to {policy_path}")
    
    print("\nPhase 6 COMPLETE. Ready for Locked Test (Phase 7).")

if __name__ == "__main__": main()
