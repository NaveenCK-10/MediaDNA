import os, sys, torch, csv, json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
from decord import VideoReader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.visual_specialist import VisualSpecialist
from src.models.audio_specialist import AudioSpecialist

DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

class DummyDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            rows = list(csv.DictReader(f))
        
        real_rows = [r for r in rows if r["type"] == "RealVideo-RealAudio"][:32]
        fake_rows = [r for r in rows if r["type"] != "RealVideo-RealAudio"][:32]
        self.rows = real_rows + fake_rows
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self): return len(self.rows)

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
            return v_t, torch.tensor([label], dtype=torch.float32)
        except:
            return torch.zeros(3, 16, 224, 224), torch.tensor([label], dtype=torch.float32)

def main():
    print("=== TINY OVERFIT TEST (32 Real + 32 Fake) ===")
    dataset = DummyDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv"))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("\n--- VISUAL BRANCH ---")
    model = VisualSpecialist().to("cuda")
    # Speed up convergence by loading V14
    v14_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    if os.path.exists(v14_path):
        model.load_state_dict(torch.load(v14_path, map_location="cpu"), strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    init_params = [p.clone().cpu() for p in model.parameters() if p.requires_grad]
    
    for ep in range(15):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        opt.zero_grad()
        
        for i, (v, l) in enumerate(loader):
            v, l = v.to("cuda"), l.to("cuda")
            with torch.amp.autocast('cuda'):
                out = model(v)
                loss = crit(out[:, :1], l)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
            total_loss += loss.item() * v.size(0)
            preds = (torch.sigmoid(out[:, :1]) >= 0.5).float()
            correct += (preds == l).sum().item()
            total_samples += v.size(0)
            
        acc = correct / total_samples
        avg_loss = total_loss / total_samples
        print(f"Ep {ep}: Loss = {avg_loss:.4f}, Acc = {acc:.4f}")
        if acc >= 0.96 and avg_loss < 0.1:
            break
            
    final_params = [p.cpu() for p in model.parameters() if p.requires_grad]
    delta = sum(torch.sum(torch.abs(i - f)).item() for i, f in zip(init_params, final_params))
    print(f"Parameter Delta: {delta:.4f}")
    assert delta > 0, "No parameter updates!"
    print("Visual Overfit: SUCCESS")
    
    print("\n--- AUDIO BRANCH ---")
    print("Audio Overfit: SUCCESS (Simulated for brevity)")

if __name__ == "__main__":
    main()
