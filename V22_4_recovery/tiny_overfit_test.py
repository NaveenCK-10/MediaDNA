"""
V22.4 Phase 2: Tiny Overfit Test (OOM-safe version)
Trains VisualSpecialist on 32 real + 32 fake balanced samples
to verify the architecture can actually learn.
Uses batch_size=2 + gradient accumulation + mixed precision.
"""
import os, sys, csv, json, time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.visual_specialist import VisualSpecialist
import torchvision.transforms as T
from decord import VideoReader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

class BalancedTinyDataset(Dataset):
    def __init__(self, manifest_path, n_per_class=32):
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        
        real_rows = [r for r in all_rows if r["type"] == "RealVideo-RealAudio"]
        fake_rows = [r for r in all_rows if r["type"] != "RealVideo-RealAudio"]
        
        np.random.seed(42)
        real_sample = [real_rows[i] for i in np.random.choice(len(real_rows), min(n_per_class, len(real_rows)), replace=False)]
        fake_sample = [fake_rows[i] for i in np.random.choice(len(fake_rows), min(n_per_class, len(fake_rows)), replace=False)]
        
        self.rows = real_sample + fake_sample
        np.random.shuffle(self.rows)
        
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Tiny dataset: {len(real_sample)} real + {len(fake_sample)} fake = {len(self.rows)} total")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        fname = row["path"]
        dir_col = row.get("", "") or [v for k,v in row.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        
        rel_dir = dir_col.replace("FakeAVCeleb/", "")
        full_path = os.path.join(DATASET_DIR, rel_dir, fname)
        cat = row["type"]
        label = 0 if cat == "RealVideo-RealAudio" else 1
        
        try:
            vr = VideoReader(full_path, width=224, height=224)
            num_frames = len(vr)
            frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_tensor = self.transform(v_tensor).permute(1, 0, 2, 3)
        except Exception as e:
            print(f"  WARNING: Failed to load {full_path}: {e}")
            v_tensor = torch.zeros(3, 16, 224, 224)
            
        return v_tensor, torch.tensor([label], dtype=torch.float32)


def main():
    print("=" * 60)
    print("V22.4 PHASE 2: TINY OVERFIT TEST (OOM-safe)")
    print("=" * 60)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    manifest = os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv")
    dataset = BalancedTinyDataset(manifest, n_per_class=32)
    # batch_size=2 to avoid OOM, with gradient accumulation
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    model = VisualSpecialist().to(DEVICE)
    
    # No pretrained weight loading - test pure architecture capability
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Record initial state
    initial_params = {n: p.clone().detach().cpu() for n, p in model.named_parameters()}
    
    ACCUM_STEPS = 4  # Effective batch size = 2 * 4 = 8
    epochs = 50
    results = []
    
    print(f"\nTraining for {epochs} epochs on {len(dataset)} samples...")
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: 2, Gradient accumulation: {ACCUM_STEPS} (effective: {2 * ACCUM_STEPS})")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        n_batches = 0
        
        optimizer.zero_grad()
        
        for step, (v, l) in enumerate(loader):
            v, l = v.to(DEVICE), l.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(v)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            # Track predictions (unscaled loss for logging)
            with torch.no_grad():
                preds = (torch.sigmoid(out[:, 0]) > 0.5).long()
                correct += (preds == l[:, 0].long()).sum().item()
                total += l.size(0)
                epoch_loss += loss.item() * ACCUM_STEPS
                n_batches += 1
            
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                # Track gradient norms before step
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Free GPU memory
            del v, l, out, loss
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        
        results.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": accuracy,
            "grad_norm": avg_grad_norm
        })
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.6f}  Acc={accuracy:.4f}  GradNorm={avg_grad_norm:.4f}")
    
    # Compute parameter delta
    param_delta = 0.0
    param_total = 0.0
    for name, p in model.named_parameters():
        delta = (p.cpu() - initial_params[name]).norm().item()
        param_delta += delta
        param_total += p.cpu().norm().item()
    
    # Final predictions on entire tiny set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for v, l in loader:
            v = v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(v)
            probs = torch.sigmoid(out[:, 0]).cpu().float().numpy()
            all_preds.extend(probs)
            all_labels.extend(l[:, 0].numpy())
            del v, out
            torch.cuda.empty_cache()
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    real_preds = all_preds[all_labels == 0]
    fake_preds = all_preds[all_labels == 1]
    
    print("\n" + "=" * 60)
    print("TINY OVERFIT TEST RESULTS")
    print("=" * 60)
    print(f"Initial Loss:     {results[0]['loss']:.6f}")
    print(f"Final Loss:       {results[-1]['loss']:.6f}")
    print(f"Loss Reduction:   {results[0]['loss'] - results[-1]['loss']:.6f}")
    print(f"Initial Accuracy: {results[0]['accuracy']:.4f}")
    print(f"Final Accuracy:   {results[-1]['accuracy']:.4f}")
    print(f"Parameter Delta:  {param_delta:.6f}")
    print(f"Parameter Total:  {param_total:.6f}")
    print(f"Delta Ratio:      {param_delta / (param_total + 1e-12):.6f}")
    print(f"Mean Grad Norm:   {np.mean([r['grad_norm'] for r in results]):.6f}")
    print(f"\nReal predictions:     mean={real_preds.mean():.4f}  std={real_preds.std():.4f}")
    print(f"Fake predictions:     mean={fake_preds.mean():.4f}  std={fake_preds.std():.4f}")
    print(f"Separation (gap):     {fake_preds.mean() - real_preds.mean():.4f}")
    
    # PASS/FAIL determination
    final_loss = results[-1]['loss']
    final_acc = results[-1]['accuracy']
    loss_reduced = results[0]['loss'] - final_loss > 0.1
    acc_high = final_acc > 0.85
    separation = fake_preds.mean() - real_preds.mean() > 0.1
    
    passed = loss_reduced or acc_high  # Either signal shows learning
    
    print(f"\n{'PASS' if passed else 'FAIL'}: loss_reduced={loss_reduced}, acc_high={acc_high}, separation={separation}")
    
    # Save results
    output = {
        "phase": "tiny_overfit_test",
        "n_real": int(sum(all_labels == 0)),
        "n_fake": int(sum(all_labels == 1)),
        "initial_loss": results[0]['loss'],
        "final_loss": final_loss,
        "initial_accuracy": results[0]['accuracy'],
        "final_accuracy": final_acc,
        "param_delta": param_delta,
        "param_delta_ratio": param_delta / (param_total + 1e-12),
        "real_pred_mean": float(real_preds.mean()),
        "fake_pred_mean": float(fake_preds.mean()),
        "separation": float(fake_preds.mean() - real_preds.mean()),
        "passed": passed,
        "all_epochs": results
    }
    
    out_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "tiny_overfit_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Clean up GPU
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
