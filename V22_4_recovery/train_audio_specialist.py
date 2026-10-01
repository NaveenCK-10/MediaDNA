"""
V22.4 Phase 5: Audio Specialist Retraining
Same fixes as visual: balanced sampling, proper loss, staged unfreezing.
"""
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.audio_specialist import AudioSpecialist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"


class AudioDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        
        self.labels = []
        for row in self.rows:
            cat = row["type"]
            label = 0 if cat == "RealVideo-RealAudio" else 1
            self.labels.append(label)
        
        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        
        weight_real = 1.0 / max(n_real, 1)
        weight_fake = 1.0 / max(n_fake, 1)
        self.sample_weights = [weight_real if l == 0 else weight_fake for l in self.labels]
        
        print(f"Audio Dataset: {n_real} real + {n_fake} fake = {len(self.rows)} total")

    def __len__(self):
        return len(self.rows)

    def _extract_mel(self, video_path):
        temp_wav = f"temp_aud_train_{os.getpid()}_{id(self)}.wav"
        try:
            cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                   "-ar", "16000", "-ac", "1", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            
            if not os.path.exists(temp_wav):
                return torch.zeros(1024, 128)
            
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
                pad = target_len - mel_spec.shape[1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
            else:
                mel_spec = mel_spec[:, :target_len]
            
            mel_spec = mel_spec.transpose(0, 1)  # (1024, 128)
            
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return mel_spec
        except Exception as e:
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return torch.zeros(1024, 128)

    def __getitem__(self, idx):
        row = self.rows[idx]
        fname = row["path"]
        dir_col = row.get("", "") or [v for k,v in row.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        
        rel_dir = dir_col.replace("FakeAVCeleb/", "")
        full_path = os.path.join(DATASET_DIR, rel_dir, fname)
        cat = row["type"]
        label = 0 if cat == "RealVideo-RealAudio" else 1
        
        cond = "RVRA"
        if cat == "RealVideo-FakeAudio": cond = "RVFA"
        elif cat == "FakeVideo-RealAudio": cond = "FVRA"
        elif cat == "FakeVideo-FakeAudio": cond = "FVFA"
        
        mel = self._extract_mel(full_path)
        return mel, torch.tensor([label], dtype=torch.float32), cond, full_path


def evaluate(model, dataloader):
    model.eval()
    preds, labels, conditions, paths = [], [], [], []
    with torch.no_grad():
        for a, l, c, p in dataloader:
            a = a.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
            conditions.extend(c)
            paths.extend(p)
            del a, out
            torch.cuda.empty_cache()
    return np.array(preds), np.array(labels), conditions, paths


def compute_full_metrics(labels, preds, threshold=0.5):
    p_bin = (preds >= threshold).astype(int)
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0
    pr_auc = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0
    bacc = balanced_accuracy_score(labels, p_bin)
    mcc = matthews_corrcoef(labels, p_bin)
    prec = precision_score(labels, p_bin, zero_division=0)
    rec = recall_score(labels, p_bin, zero_division=0)
    f1 = f1_score(labels, p_bin, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(labels, p_bin).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    except:
        spec = 0
    return {
        "ROC_AUC": auc, "PR_AUC": pr_auc, "balanced_accuracy": bacc,
        "MCC": mcc, "precision": prec, "recall": rec, "specificity": spec, "F1": f1
    }


def main():
    print("=" * 60)
    print("V22.4 PHASE 5: AUDIO SPECIALIST RETRAINING")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    
    train_dataset = AudioDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv"))
    dev_dataset = AudioDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
    
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=800,
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model = AudioSpecialist().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    ACCUM_STEPS = 2
    
    total = sum(p.numel() for p in model.parameters())
    
    # ============ STAGE A: Head-only ============
    print("\n--- STAGE A: Freeze encoder, train head only ---")
    for name, param in model.named_parameters():
        if 'audio_encoder' in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total:,}")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    best_mcc = -1.0
    ckpt_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4C_AUDIO_CHECKPOINT.pth")
    run_manifest = []
    
    STAGE_EPOCHS = 5
    patience = 3
    patience_counter = 0
    
    for epoch in range(STAGE_EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for i, (a, l, _, _) in enumerate(train_loader):
            a, l = a.to(DEVICE), l.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(a)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUM_STEPS
            n_batches += 1
            del a, l, out, loss
            torch.cuda.empty_cache()
        
        avg_loss = train_loss / max(n_batches, 1)
        preds, labels, _, _ = evaluate(model, dev_loader)
        
        best_t_mcc = -1
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            p_bin = (preds >= t).astype(int)
            t_mcc = matthews_corrcoef(labels, p_bin)
            if t_mcc > best_t_mcc:
                best_t_mcc = t_mcc
                best_t = t
        
        metrics = compute_full_metrics(labels, preds, threshold=best_t)
        print(f"  StageA Ep {epoch+1}: Loss={avg_loss:.4f}  AUC={metrics['ROC_AUC']:.4f}  MCC={metrics['MCC']:.4f}  BAcc={metrics['balanced_accuracy']:.4f}")
        
        run_manifest.append({"stage": "A", "epoch": epoch+1, "loss": avg_loss, **metrics})
        
        if metrics['MCC'] > best_mcc:
            best_mcc = metrics['MCC']
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> New best MCC={best_mcc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    -> Early stopping")
                break
    
    # ============ STAGE B: Unfreeze last 2 blocks ============
    print(f"\n--- STAGE B: Unfreeze last 2 encoder blocks ---")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(DEVICE)
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'mlp_audio' in name or 'mlp_head' in name:
            param.requires_grad = True
        if 'audio_encoder.transformer.10.' in name or 'audio_encoder.transformer.11.' in name:
            param.requires_grad = True
        if 'audio_encoder.norm.' in name:
            param.requires_grad = True
    
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and 'audio_encoder' in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'audio_encoder' not in n]
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 5e-4}
    ])
    
    patience_counter = 0
    for epoch in range(STAGE_EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for i, (a, l, _, _) in enumerate(train_loader):
            a, l = a.to(DEVICE), l.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(a)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUM_STEPS
            n_batches += 1
            del a, l, out, loss
            torch.cuda.empty_cache()
        
        avg_loss = train_loss / max(n_batches, 1)
        preds, labels, _, _ = evaluate(model, dev_loader)
        
        best_t_mcc = -1
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            p_bin = (preds >= t).astype(int)
            t_mcc = matthews_corrcoef(labels, p_bin)
            if t_mcc > best_t_mcc:
                best_t_mcc = t_mcc
                best_t = t
        
        metrics = compute_full_metrics(labels, preds, threshold=best_t)
        print(f"  StageB Ep {epoch+1}: Loss={avg_loss:.4f}  AUC={metrics['ROC_AUC']:.4f}  MCC={metrics['MCC']:.4f}")
        
        run_manifest.append({"stage": "B", "epoch": epoch+1, "loss": avg_loss, **metrics})
        
        if metrics['MCC'] > best_mcc:
            best_mcc = metrics['MCC']
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> New best MCC={best_mcc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Final eval
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(DEVICE)
    preds, labels, conditions, paths = evaluate(model, dev_loader)
    
    best_t = 0.5
    best_t_mcc = -1
    for t in np.arange(0.01, 0.99, 0.01):
        p_bin = (preds >= t).astype(int)
        t_mcc = matthews_corrcoef(labels, p_bin)
        if t_mcc > best_t_mcc:
            best_t_mcc = t_mcc
            best_t = t
    
    final_metrics = compute_full_metrics(labels, preds, threshold=best_t)
    print(f"\nFinal V22.4 Audio Specialist DEV Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "V22.4 Audio Specialist",
        "optimal_threshold": float(best_t),
        "best_dev_MCC": float(best_mcc),
        "final_metrics": final_metrics,
        "run_manifest": run_manifest
    }
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4C_AUDIO_RESULTS.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    sha = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha.update(block)
    with open(ckpt_path + ".sha256", "w") as f:
        f.write(sha.hexdigest())
    
    print(f"Checkpoint: {ckpt_path}")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
