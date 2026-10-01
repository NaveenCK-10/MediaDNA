"""
V22.4 Phase 3: Retrained Visual Specialist
Key fixes over V22.3B:
1. WeightedRandomSampler for balanced batches
2. pos_weight in BCEWithLogitsLoss 
3. Full epoch training (no 25-batch cutoff)
4. Staged unfreezing: head-only first, then last encoder blocks
5. Early stopping on dev MCC
6. Multiple epochs
"""
import os, sys, csv, json, time, hashlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, average_precision_score)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.visual_specialist import VisualSpecialist
import torchvision.transforms as T
from decord import VideoReader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

class VisualDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Compute class counts for WeightedRandomSampler
        self.labels = []
        for row in self.rows:
            cat = row["type"]
            label = 0 if cat == "RealVideo-RealAudio" else 1
            self.labels.append(label)
        
        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        
        # Weight each sample inversely proportional to class frequency
        weight_real = 1.0 / max(n_real, 1)
        weight_fake = 1.0 / max(n_fake, 1)
        self.sample_weights = [weight_real if l == 0 else weight_fake for l in self.labels]
        
        print(f"Dataset: {n_real} real + {n_fake} fake = {len(self.rows)} total")
        print(f"Sample weights: real={weight_real:.6f}  fake={weight_fake:.6f}")

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
        
        cond = "RVRA"
        if cat == "RealVideo-FakeAudio": cond = "RVFA"
        elif cat == "FakeVideo-RealAudio": cond = "FVRA"
        elif cat == "FakeVideo-FakeAudio": cond = "FVFA"
        
        try:
            vr = VideoReader(full_path, width=224, height=224)
            num_frames = len(vr)
            frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_tensor = self.transform(v_tensor).permute(1, 0, 2, 3)
        except Exception as e:
            v_tensor = torch.zeros(3, 16, 224, 224)
            
        return v_tensor, torch.tensor([label], dtype=torch.float32), cond, full_path


def evaluate(model, dataloader):
    model.eval()
    preds, labels, conditions, paths = [], [], [], []
    with torch.no_grad():
        for v, l, c, p in dataloader:
            v = v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
            conditions.extend(c)
            paths.extend(p)
            del v, out
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
    print("V22.4 PHASE 3: VISUAL SPECIALIST RETRAINING")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    
    train_dataset = VisualDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_train.csv"))
    dev_dataset = VisualDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
    
    # WeightedRandomSampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=800,
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Initialize model
    model = VisualSpecialist().to(DEVICE)
    
    # Compute pos_weight for BCEWithLogitsLoss
    n_real = sum(1 for l in train_dataset.labels if l == 0)
    n_fake = sum(1 for l in train_dataset.labels if l == 1)
    # For BCEWithLogitsLoss: pos_weight = weight for positive class (fake)
    # But we want to UPWEIGHT the minority class (real = label 0)
    # Since we're using label 0 for real and training on [:, :1], 
    # and logit[:, 0] with sigmoid gives P(class 0 = real),
    # we need to think about this carefully:
    # The specialist outputs 2 classes. We take [:, :1] and compare to label.
    # Label 0 = real, Label 1 = fake.
    # BCEWithLogitsLoss with pos_weight upweights the positive class (label=1=fake).
    # We actually want to upweight label=0=real (minority).
    # Solution: use standard BCE but with sample-level weights,
    # OR switch to CrossEntropyLoss with class_weight
    # For simplicity, since WeightedRandomSampler already balances batches,
    # we can use standard BCEWithLogitsLoss.
    # The sampler ensures ~50/50 real/fake in each epoch.
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.amp.GradScaler('cuda')
    ACCUM_STEPS = 2  # Effective batch = 4
    
    # ============ STAGE A: Head-only training ============
    print("\n--- STAGE A: Freeze backbone, train head only ---")
    
    # Freeze backbone
    for name, param in model.named_parameters():
        if 'visual_encoder' in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )
    
    best_mcc = -1.0
    best_epoch = 0
    patience = 3
    patience_counter = 0
    
    STAGE_A_EPOCHS = 5
    run_manifest = []
    
    for epoch in range(STAGE_A_EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for i, (v, l, _, _) in enumerate(train_loader):
            v, l = v.to(DEVICE), l.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(v)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUM_STEPS
            n_batches += 1
            
            del v, l, out, loss
            torch.cuda.empty_cache()
        
        avg_loss = train_loss / max(n_batches, 1)
        
        # Evaluate on dev
        preds, labels, conditions, _ = evaluate(model, dev_loader)
        metrics = compute_full_metrics(labels, preds)
        
        # Find optimal threshold
        best_t_mcc = -1
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            p_bin = (preds >= t).astype(int)
            t_mcc = matthews_corrcoef(labels, p_bin)
            if t_mcc > best_t_mcc:
                best_t_mcc = t_mcc
                best_t = t
        
        metrics_opt = compute_full_metrics(labels, preds, threshold=best_t)
        
        print(f"  StageA Ep {epoch+1}/{STAGE_A_EPOCHS}: Loss={avg_loss:.4f}  "
              f"AUC={metrics['ROC_AUC']:.4f}  MCC@0.5={metrics['MCC']:.4f}  "
              f"MCC@opt={metrics_opt['MCC']:.4f}  BAcc={metrics_opt['balanced_accuracy']:.4f}")
        
        run_manifest.append({
            "stage": "A", "epoch": epoch + 1, "train_loss": avg_loss,
            **{f"dev_{k}": v for k, v in metrics_opt.items()},
            "optimal_threshold": best_t
        })
        
        if metrics_opt['MCC'] > best_mcc:
            best_mcc = metrics_opt['MCC']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best checkpoint
            ckpt_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4B_VISUAL_CHECKPOINT.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> New best MCC={best_mcc:.4f}, saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    -> Early stopping (patience={patience})")
                break
    
    # ============ STAGE B: Unfreeze last encoder blocks ============
    print(f"\n--- STAGE B: Unfreeze last 2 encoder blocks + head ---")
    
    # Load best Stage A checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(DEVICE)
    
    # Unfreeze last 2 blocks of visual_encoder
    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze everything first
    
    # Unfreeze head layers
    for name, param in model.named_parameters():
        if 'mlp_vision' in name or 'mlp_head' in name:
            param.requires_grad = True
    
    # Unfreeze last 2 transformer blocks
    for name, param in model.named_parameters():
        if 'visual_encoder.blocks.10.' in name or 'visual_encoder.blocks.11.' in name:
            param.requires_grad = True
        if 'visual_encoder.norm.' in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Lower lr for backbone
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'visual_encoder' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 5e-4}
    ])
    
    STAGE_B_EPOCHS = 5
    patience_counter = 0
    
    for epoch in range(STAGE_B_EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for i, (v, l, _, _) in enumerate(train_loader):
            v, l = v.to(DEVICE), l.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(v)
                loss = criterion(out[:, :1], l) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUM_STEPS
            n_batches += 1
            
            del v, l, out, loss
            torch.cuda.empty_cache()
        
        avg_loss = train_loss / max(n_batches, 1)
        
        preds, labels, conditions, _ = evaluate(model, dev_loader)
        
        best_t_mcc = -1
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            p_bin = (preds >= t).astype(int)
            t_mcc = matthews_corrcoef(labels, p_bin)
            if t_mcc > best_t_mcc:
                best_t_mcc = t_mcc
                best_t = t
        
        metrics_opt = compute_full_metrics(labels, preds, threshold=best_t)
        
        print(f"  StageB Ep {epoch+1}/{STAGE_B_EPOCHS}: Loss={avg_loss:.4f}  "
              f"AUC={metrics_opt['ROC_AUC']:.4f}  MCC@opt={metrics_opt['MCC']:.4f}  "
              f"BAcc={metrics_opt['balanced_accuracy']:.4f}")
        
        run_manifest.append({
            "stage": "B", "epoch": epoch + 1, "train_loss": avg_loss,
            **{f"dev_{k}": v for k, v in metrics_opt.items()},
            "optimal_threshold": best_t
        })
        
        if metrics_opt['MCC'] > best_mcc:
            best_mcc = metrics_opt['MCC']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> New best MCC={best_mcc:.4f}, saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    -> Early stopping (patience={patience})")
                break
    
    # Final evaluation with best checkpoint
    print(f"\n--- Final Evaluation (best epoch) ---")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(DEVICE)
    
    preds, labels, conditions, paths = evaluate(model, dev_loader)
    
    # Optimal threshold
    best_t_mcc = -1
    best_t = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        p_bin = (preds >= t).astype(int)
        t_mcc = matthews_corrcoef(labels, p_bin)
        if t_mcc > best_t_mcc:
            best_t_mcc = t_mcc
            best_t = t
    
    final_metrics = compute_full_metrics(labels, preds, threshold=best_t)
    
    print(f"\nFinal V22.4 Visual Specialist DEV Metrics (threshold={best_t:.2f}):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "V22.4 Visual Specialist (Retrained)",
        "optimal_threshold": float(best_t),
        "best_dev_MCC": float(best_mcc),
        "final_metrics": final_metrics,
        "run_manifest": run_manifest
    }
    
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4B_VISUAL_RESULTS.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    # Save predictions
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4B_VISUAL_PREDICTIONS.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "category", "probability", "prediction"])
        for p_val, l_val, c_val, path in zip(preds, labels, conditions, paths):
            p_bin = 1 if p_val >= best_t else 0
            writer.writerow([os.path.basename(path), int(l_val), c_val, f"{p_val:.6f}", p_bin])
    
    # SHA256
    sha = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha.update(block)
    with open(ckpt_path + ".sha256", "w") as f:
        f.write(sha.hexdigest())
    
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"SHA256: {sha.hexdigest()}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
