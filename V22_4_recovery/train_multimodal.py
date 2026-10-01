import os, sys, csv, json, time, hashlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, average_precision_score)
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
        
        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        weight_real = 1.0 / max(n_real, 1)
        weight_fake = 1.0 / max(n_fake, 1)
        self.sample_weights = [weight_real if l == 0 else weight_fake for l in self.labels]
        print(f"Multimodal Dataset: {n_real} real + {n_fake} fake = {len(self.rows)} total")

    def __len__(self): return len(self.rows)

    def _extract_mel(self, video_path):
        temp_wav = f"temp_aud_{os.getpid()}_{id(self)}.wav"
        try:
            cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                   "-ar", "16000", "-ac", "1", temp_wav]
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
        cond = {"RealVideo-RealAudio": "RVRA", "RealVideo-FakeAudio": "RVFA", "FakeVideo-RealAudio": "FVRA", "FakeVideo-FakeAudio": "FVFA"}.get(r["type"], "RVRA")
        
        try:
            vr = VideoReader(full_path, width=224, height=224)
            frame_idx = np.linspace(0, len(vr) - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_t = self.transform(v_t).permute(1, 0, 2, 3)
        except:
            v_t = torch.zeros(3, 16, 224, 224)
            
        a_t = self._extract_mel(full_path)
        return a_t, v_t, torch.tensor([label], dtype=torch.float32), cond, full_path

def evaluate(model, dataloader):
    model.eval()
    preds, labels, conditions, paths = [], [], [], []
    with torch.no_grad():
        for a, v, l, c, p in dataloader:
            a, v = a.to(DEVICE), v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                # VideoCAVMAEFT expects audio (B, 1024, 128) and video (B, 3, 16, 224, 224)
                # Note: our forward is out = model(a_t.unsqueeze(1), v_t.unsqueeze(0)) wait no
                # In VideoCAVMAEFT: forward(audio, visual)
                # audio: (B, 1024, 128) wait, the original train code uses a.unsqueeze(1) for channel?
                # Let's check how AVFF inference passes it: model(a_t.unsqueeze(0), v_t.unsqueeze(0))
                # Our dataloader already returns (B, 1024, 128) and (B, 3, 16, 224, 224)
                # Actually AudioSpecialist uses unsqueeze(1) internally. AVFF uses it directly.
                # In VideoCAVMAEFT: audio shape (B, 1, 1024, 128) or (B, 1024, 128)?
                if len(a.shape) == 3: a = a.unsqueeze(1)
                out = model(a, v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
            conditions.extend(c)
            paths.extend(p)
    return np.array(preds), np.array(labels), conditions, paths

def main():
    print("=" * 60)
    print("V22.4 PHASE 6: MULTIMODAL FINE-TUNING")
    print("=" * 60)
    
    train_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv"))
    dev_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
    
    sampler = WeightedRandomSampler(train_dataset.sample_weights, 800, True)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    model = VideoCAVMAEFT()
    v14_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    if os.path.exists(v14_path):
        model.load_state_dict(torch.load(v14_path, map_location="cpu"), strict=False)
    model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    ACCUM_STEPS = 2
    
    print("\n--- STAGE A: Freeze encoders, train cross-modal + head ---")
    for name, param in model.named_parameters(): param.requires_grad = False
    for name, param in model.named_parameters():
        if 'mlp_head' in name or 'mlp_vision' in name or 'mlp_audio' in name or 'vision_proj' in name or 'audio_proj' in name:
            param.requires_grad = True
            
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    best_mcc = -1.0
    ckpt_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4F_MULTIMODAL_CHECKPOINT.pth")
    
    for epoch in range(2): # Only 2 epochs per stage to ensure it completes
        model.train()
        train_loss = 0
        opt.zero_grad()
        for i, (a, v, l, _, _) in enumerate(train_loader):
            a, v, l = a.to(DEVICE), v.to(DEVICE), l.to(DEVICE)
            if len(a.shape) == 3: a = a.unsqueeze(1)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            train_loss += loss.item() * ACCUM_STEPS
        
        preds, labels, _, _ = evaluate(model, dev_loader)
        p_bin = (preds >= 0.5).astype(int)
        mcc = matthews_corrcoef(labels, p_bin)
        print(f"StageA Ep {epoch+1}: Loss {train_loss/(i+1):.4f}, MCC {mcc:.4f}")
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__": main()
