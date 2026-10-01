import os, sys, csv, time, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    def __init__(self, manifest_path, subset_size=None):
        with open(manifest_path, 'r') as f:
            self.rows = list(csv.DictReader(f))
        
        if subset_size:
            self.rows = self.rows[:subset_size]
            
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.labels = []
        for r in self.rows:
            self.labels.append(0 if r["type"] == "RealVideo-RealAudio" else 1)
        
        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        weight_real = 1.0 / max(n_real, 1)
        weight_fake = 1.0 / max(n_fake, 1)
        self.sample_weights = [weight_real if l == 0 else weight_fake for l in self.labels]
        print(f"Dataset Loaded: {n_real} real + {n_fake} fake = {len(self.rows)} total")

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

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for i, (a, v, l, _) in enumerate(dataloader):
            if i % 100 == 0: print(f"DEV progress: {i}/{len(dataloader)} batches")
            a, v = a.to(DEVICE), v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
            
    p_bin = (np.array(preds) >= 0.5).astype(int)
    labels = np.array(labels)
    try: auc = roc_auc_score(labels, preds)
    except: auc = 0.5
    bacc = balanced_accuracy_score(labels, p_bin)
    mcc = matthews_corrcoef(labels, p_bin)
    return preds, labels, auc, bacc, mcc

def run_sanity_test(model, loader):
    print("\n" + "="*50)
    print("TASK 4: SANITY & OVERFIT TEST")
    print("="*50)
    
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    model.train()
    a, v, l, _ = next(iter(loader))
    a, v, l = a.to(DEVICE), v.to(DEVICE), l.to(DEVICE)
    
    print(f"Target Labels: {l.cpu().numpy().tolist()}")
    
    p_cross_initial = model.a2v.mlp.linear.weight.clone().detach()
    p_head_initial = model.mlp_head.fc1.weight.clone().detach()
    
    for ep in range(5):
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(a, v)
            loss = criterion(out[:, :1], l)
        loss.backward()
        opt.step()
        prob = torch.sigmoid(out)[:, 0].detach().cpu().numpy()
        print(f"  Sanity Ep {ep}: Loss={loss.item():.4f}, Probs={prob}")
        
    p_cross_final = model.a2v.mlp.linear.weight.clone().detach()
    p_head_final = model.mlp_head.fc1.weight.clone().detach()
    diff_cross = torch.norm(p_cross_final - p_cross_initial).item()
    diff_head = torch.norm(p_head_final - p_head_initial).item()
    
    print(f"  Cross-modal parameter diff: {diff_cross:.6f}")
    print(f"  MLP Head parameter diff: {diff_head:.6f}")
    
    if diff_cross == 0 or diff_head == 0 or loss.item() > 0.69:
        print("\nCRITICAL FAILURE: Sanity test failed! Parameters didn't change or loss didn't decrease.")
        sys.exit(1)
    print("Sanity overfit test PASSED.\n")

def apply_stage_freeze_policy(model, stage):
    for param in model.parameters(): param.requires_grad = False
    unfrozen_names = []
    
    for name, param in model.named_parameters():
        if stage == 'A':
            if 'mlp_head' in name or 'mlp_vision' in name or 'mlp_audio' in name or 'a2v' in name or 'v2a' in name:
                param.requires_grad = True
                unfrozen_names.append(name)
        elif stage == 'B':
            if 'mlp_head' in name or 'mlp_vision' in name or 'mlp_audio' in name or 'a2v' in name or 'v2a' in name:
                param.requires_grad = True
                unfrozen_names.append(name)
            if 'visual_encoder.blocks.10' in name or 'visual_encoder.blocks.11' in name:
                param.requires_grad = True
                unfrozen_names.append(name)
            if 'audio_encoder.transformer.10' in name or 'audio_encoder.transformer.11' in name:
                param.requires_grad = True
                unfrozen_names.append(name)

    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"\n--- STAGE {stage} FREEZE POLICY ---")
    print(f"Trainable layers: {trainable_params}, Frozen layers: {frozen_params}")
    print("Representative Trainable Layers:", unfrozen_names[:5] + ["..."] + unfrozen_names[-5:])

def train_stage(model, stage, epochs, train_loader, save_dir):
    apply_stage_freeze_policy(model, stage)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3 if stage == 'A' else 1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    ACCUM_STEPS = 2
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0
        opt.zero_grad()
        
        preds, labels = [], []
        
        # Track parameters before epoch to confirm they changed
        p_initial = model.mlp_head.fc1.weight.clone().detach()
        
        for i, (a, v, l, _) in enumerate(train_loader):
            a, v, l = a.to(DEVICE), v.to(DEVICE), l.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            scaler.scale(loss).backward()
            
            prob = torch.sigmoid(out)[:, 0].detach().cpu().numpy()
            preds.extend(prob)
            labels.extend(l.cpu().numpy()[:, 0])
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                
            train_loss += loss.item() * ACCUM_STEPS
            if (i+1) % 50 == 0: print(f"  Batch {i+1}/{len(train_loader)} Loss: {loss.item()*ACCUM_STEPS:.4f}")
            
        p_final = model.mlp_head.fc1.weight.clone().detach()
        param_changed = torch.norm(p_final - p_initial).item() > 0
        
        cross_grad, head_grad = 0.0, 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                if 'a2v' in name or 'v2a' in name: cross_grad = p.grad.norm().item()
                if 'mlp_head.fc1' in name: head_grad = p.grad.norm().item()
                
        p_bin = (np.array(preds) >= 0.5).astype(int)
        n_pred_real = sum(1 for p in p_bin if p == 0)
        n_pred_fake = sum(1 for p in p_bin if p == 1)
        
        ckpt_path = os.path.join(save_dir, f"V22_4F_MULTIMODAL_Stage{stage}_Ep{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        
        duration = time.time() - t0
        print(f"\n[DONE] Stage {stage} Epoch {epoch+1} - Duration: {duration:.1f} sec")
        print(f"  Loss: {train_loss/(i+1):.4f} | Batches: {i+1} | Samples: {(i+1)*2}")
        print(f"  Train Preds: {n_pred_real} Real, {n_pred_fake} Fake | Mean Prob={np.mean(preds):.4f}")
        print(f"  Gradients: CrossModal={cross_grad:.4f}, Head={head_grad:.4f}")
        print(f"  Trainable params changed: {param_changed}")
        print(f"  Checkpoint saved: {ckpt_path}")

def main():
    print("=" * 60)
    print("V22.4 MULTIMODAL REPAIR - FAST TRAINING MODE")
    print("=" * 60)
    
    # 1. Print intended checkpoints
    save_dir = os.path.join(PROJECT_ROOT, "V22_4_recovery")
    print("\nIntended Checkpoints:")
    print("  " + os.path.join(save_dir, "V22_4F_MULTIMODAL_StageA_Ep1.pth"))
    print("  " + os.path.join(save_dir, "V22_4F_MULTIMODAL_StageA_Ep2.pth"))
    print("  " + os.path.join(save_dir, "V22_4F_MULTIMODAL_StageB_Ep1.pth"))
    print("  " + os.path.join(save_dir, "V22_4F_MULTIMODAL_StageB_Ep2.pth"))
    
    train_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv"))
    dev_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
    
    sampler = WeightedRandomSampler(train_dataset.sample_weights, 800, True)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    model = VideoCAVMAEFT()
    
    v14_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    if os.path.exists(v14_path):
        model.load_state_dict(torch.load(v14_path, map_location="cpu"), strict=False)
        print("Loaded V14 base weights.")
        
    visual_ckpt = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4B_VISUAL_CHECKPOINT.pth")
    if os.path.exists(visual_ckpt):
        v_sd = torch.load(visual_ckpt, map_location="cpu")
        v_sd = {k: v for k, v in v_sd.items() if k.startswith("visual_encoder.")}
        res = model.load_state_dict(v_sd, strict=False)
        print(f"Imported V22.4 Visual Encoder: {len(v_sd)} keys.")
        
    audio_ckpt = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4C_AUDIO_CHECKPOINT.pth")
    if os.path.exists(audio_ckpt):
        a_sd = torch.load(audio_ckpt, map_location="cpu")
        a_sd = {k: v for k, v in a_sd.items() if k.startswith("audio_encoder.")}
        res = model.load_state_dict(a_sd, strict=False)
        print(f"Imported V22.4 Audio Encoder: {len(a_sd)} keys.")
        
    model.to(DEVICE)
    
    run_sanity_test(model, train_loader)
    
    # STAGE A
    print("\n--- INITIATING STAGE A (2 Epochs) ---")
    train_stage(model, 'A', 2, train_loader, save_dir)
    
    # STAGE B
    print("\n--- INITIATING STAGE B (2 Epochs) ---")
    train_stage(model, 'B', 2, train_loader, save_dir)
    
    print("\n" + "="*50)
    print("FINAL 2264-SAMPLE DEV EVALUATION (Stage B Epoch 2 Model)")
    print("="*50)
    
    t0 = time.time()
    preds, labels, auc, bacc, mcc = evaluate(model, dev_loader)
    eval_dur = time.time() - t0
    
    p_bin = (preds >= 0.5).astype(int)
    
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
