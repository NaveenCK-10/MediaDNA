import os, sys, csv, time, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
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

def main():
    print("=" * 60)
    print("V22.4 MULTIMODAL REPAIR - FINAL EVALUATION RECOVERY")
    print("=" * 60)
    
    dev_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    model = VideoCAVMAEFT()
    ckpt_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4F_MULTIMODAL_StageB_Ep2.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.to(DEVICE)
    model.eval()
    
    preds, labels = [], []
    t0 = time.time()
    
    with torch.no_grad():
        for i, (a, v, l, _) in enumerate(dev_loader):
            if i % 100 == 0: print(f"DEV progress: {i}/{len(dev_loader)} batches")
            a, v = a.to(DEVICE), v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
            
    eval_dur = time.time() - t0
    preds = np.array(preds)
    labels = np.array(labels)
    p_bin = (preds >= 0.5).astype(int)
    
    try: auc = roc_auc_score(labels, preds)
    except: auc = 0.5
    bacc = balanced_accuracy_score(labels, p_bin)
    mcc = matthews_corrcoef(labels, p_bin)
    
    print(f"\nFinal DEV Eval Duration: {eval_dur:.1f} sec")
    print(f"ROC-AUC:           {auc:.4f}")
    print(f"PR-AUC:            {average_precision_score(labels, preds):.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"Precision:         {precision_score(labels, p_bin, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(labels, p_bin, zero_division=0):.4f}")
    
    cm = confusion_matrix(labels, p_bin)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Specificity:       {tn/(tn+fp):.4f}")
    print(f"F1 Score:          {f1_score(labels, p_bin, zero_division=0):.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    n_pred_real = sum(1 for p in p_bin if p == 0)
    n_pred_fake = sum(1 for p in p_bin if p == 1)
    print(f"\nPrediction Distribution: {n_pred_real} Real / {n_pred_fake} Fake")
    print(f"Mean Predicted Probability: {np.mean(preds):.4f}")

if __name__ == "__main__": main()
