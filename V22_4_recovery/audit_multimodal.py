import os, sys, csv, time, traceback, json, copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

class AuditMultimodalDataset(Dataset):
    def __init__(self, manifest_path, subset_size=None):
        with open(manifest_path, 'r') as f:
            self.rows = list(csv.DictReader(f))
        
        if subset_size:
            self.rows = self.rows[:subset_size]
            
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.counters = {
            "requested": 0,
            "vid_success": 0, "vid_fail": 0,
            "aud_success": 0, "aud_fail": 0,
            "zero_fallback": 0,
            "vid_time": [], "aud_time": [],
            "exceptions": {}
        }
        
        self.labels = [0 if r["type"] == "RealVideo-RealAudio" else 1 for r in self.rows]
        print(f"Init Dataset: {sum(1 for l in self.labels if l==0)} Real / {sum(1 for l in self.labels if l==1)} Fake")

    def __len__(self): return len(self.rows)

    def _extract_mel(self, video_path):
        temp_wav = f"temp_aud_audit_{os.getpid()}_{id(self)}.wav"
        t0 = time.time()
        try:
            cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            if not os.path.exists(temp_wav): 
                raise FileNotFoundError(f"FFMPEG output {temp_wav} missing")
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
            
            self.counters["aud_success"] += 1
            self.counters["aud_time"].append(time.time() - t0)
            return mel_spec, True
        except Exception as e:
            if os.path.exists(temp_wav): os.remove(temp_wav)
            self.counters["aud_fail"] += 1
            err = f"AudError: {type(e).__name__} - {str(e)}"
            self.counters["exceptions"][err] = self.counters["exceptions"].get(err, 0) + 1
            return torch.zeros(1024, 128), False

    def __getitem__(self, idx):
        self.counters["requested"] += 1
        r = self.rows[idx]
        fname = r["path"]
        dir_col = r.get("", "") or [v for k,v in r.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        full_path = os.path.join(DATASET_DIR, dir_col.replace("FakeAVCeleb/", ""), fname)
        label = 0 if r["type"] == "RealVideo-RealAudio" else 1
        
        t0 = time.time()
        v_success = False
        try:
            vr = VideoReader(full_path, width=224, height=224)
            frame_idx = np.linspace(0, len(vr) - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_t = self.transform(v_t).permute(1, 0, 2, 3)
            self.counters["vid_success"] += 1
            v_success = True
        except Exception as e:
            v_t = torch.zeros(3, 16, 224, 224)
            self.counters["vid_fail"] += 1
            err = f"VidError: {type(e).__name__} - {str(e)}"
            self.counters["exceptions"][err] = self.counters["exceptions"].get(err, 0) + 1
        self.counters["vid_time"].append(time.time() - t0)
            
        a_t, a_success = self._extract_mel(full_path)
        
        if not v_success or not a_success:
            self.counters["zero_fallback"] += 1
            print(f"[FAIL] {fname} (Vid: {v_success}, Aud: {a_success})")
            
        return a_t, v_t, torch.tensor([label], dtype=torch.float32)

def run_diagnostic():
    print("="*60)
    print("=== MULTIMODAL DATA PIPELINE & TRAINING AUDIT ===")
    print("="*60)
    
    print("\\n[1] DATASET DIAGNOSTIC")
    train_ds = AuditMultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv"), subset_size=32)
    dev_ds = AuditMultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"), subset_size=32)
    
    loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    
    print("\\n[2] MODEL DIAGNOSTIC")
    model = VideoCAVMAEFT()
    
    # Initialize from V14
    v14_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    if os.path.exists(v14_path):
        model.load_state_dict(torch.load(v14_path, map_location="cpu"), strict=False)
        print("Loaded V14 base weights.")
        
    visual_ckpt = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4B_VISUAL_CHECKPOINT.pth")
    if os.path.exists(visual_ckpt):
        v_sd = torch.load(visual_ckpt, map_location="cpu")
        v_sd = {k: v for k, v in v_sd.items() if k.startswith("visual_encoder.")}
        model.load_state_dict(v_sd, strict=False)
        print(f"Loaded V22.4 Visual Specialist encoder weights. ({len(v_sd)} keys)")
        
    audio_ckpt = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4C_AUDIO_CHECKPOINT.pth")
    if os.path.exists(audio_ckpt):
        a_sd = torch.load(audio_ckpt, map_location="cpu")
        a_sd = {k: v for k, v in a_sd.items() if k.startswith("audio_encoder.")}
        model.load_state_dict(a_sd, strict=False)
        print(f"Loaded V22.4 Audio Specialist encoder weights. ({len(a_sd)} keys)")
        
    model.to(DEVICE)
    model.train()
    
    print("\\n[3] STAGE A FREEZE POLICY AUDIT")
    for name, param in model.named_parameters(): param.requires_grad = False
    for name, param in model.named_parameters():
        if 'mlp_head' in name or 'mlp_vision' in name or 'mlp_audio' in name or 'vision_proj' in name or 'audio_proj' in name:
            param.requires_grad = True
            
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"Trainable layers: {trainable_params}, Frozen layers: {frozen_params}")
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\\n[4] FORWARD/BACKWARD INFERENCE & GRADIENT AUDIT (32 TRAIN SAMPLES)")
    
    param_before = None
    param_after = None
    
    for i, (a, v, l) in enumerate(loader):
        a, v, l = a.to(DEVICE), v.to(DEVICE), l.to(DEVICE)
        
        if i == 0:
            print(f"\\nBatch 0 Tensors entering model:")
            print(f"  Audio : shape={a.shape}, min={a.min().item():.4f}, max={a.max().item():.4f}, mean={a.mean().item():.4f}")
            print(f"  Video : shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
            print(f"  Label : shape={l.shape}, values={l.cpu().numpy().tolist()} (0=Real, 1=Fake)")
            
            # Save parameter state before update
            param_before = model.mlp_head.fc1.weight.clone().detach()

        with torch.amp.autocast('cuda'):
            out = model(a, v)
            loss = criterion(out[:, :1], l)
            
        loss.backward()
        
        if i == 0:
            print("\\nGradient Norms (Stage A allowed layers):")
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if 'mlp_head.fc1.weight' in name or 'vision_proj.weight' in name:
                        print(f"  {name}: {p.grad.norm().item():.6f}")
                elif not p.requires_grad and p.grad is not None:
                    print(f"  CRITICAL ERROR: Frozen parameter {name} received gradient!")
                    
            print("\\nLogits/Probs (Batch 0):")
            prob = torch.sigmoid(out).detach()
            for j in range(len(prob)):
                print(f"  Sample {j}: Target={l[j,0].item()}, Logit={out[j,0].item():.4f}, Prob={prob[j,0].item():.4f}")
            
        opt.step()
        opt.zero_grad()
        
        if i == 0:
            param_after = model.mlp_head.fc1.weight.clone().detach()
            diff = torch.norm(param_after - param_before).item()
            print(f"\\nParameter Update Verification (mlp_head.fc1.weight):")
            print(f"  L2 diff before/after step = {diff:.6f}")
            if diff == 0:
                print("  CRITICAL ERROR: Trainable parameter did not update.")
            else:
                print("  SUCCESS: Optimizer step modified the trainable parameter.")

    print("\\n[5] TRAIN PIPELINE EXTRACTION RESULTS")
    train_ds.counters["vid_time_avg"] = np.mean(train_ds.counters["vid_time"]) if train_ds.counters["vid_time"] else 0
    train_ds.counters["aud_time_avg"] = np.mean(train_ds.counters["aud_time"]) if train_ds.counters["aud_time"] else 0
    del train_ds.counters["vid_time"], train_ds.counters["aud_time"]
    print(json.dumps(train_ds.counters, indent=2))
    
    print("\\n[6] DEV INFERENCE AUDIT (32 DEV SAMPLES)")
    dev_loader = DataLoader(dev_ds, batch_size=2)
    model.eval()
    with torch.no_grad():
        for a, v, l in dev_loader:
            pass
            
    print("\\n[7] DEV PIPELINE EXTRACTION RESULTS")
    dev_ds.counters["vid_time_avg"] = np.mean(dev_ds.counters["vid_time"]) if dev_ds.counters["vid_time"] else 0
    dev_ds.counters["aud_time_avg"] = np.mean(dev_ds.counters["aud_time"]) if dev_ds.counters["aud_time"] else 0
    del dev_ds.counters["vid_time"], dev_ds.counters["aud_time"]
    print(json.dumps(dev_ds.counters, indent=2))
    
if __name__ == "__main__":
    run_diagnostic()
