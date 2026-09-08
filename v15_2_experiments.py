import os
import csv
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader

# Configuration
CHECKPOINT = "exp/stage-3-local/models/best_audio_model.pth"
VAL_CSV = "data/val_v14.csv"
TRAIN_CSV = "data/train_v14.csv"
TEST_LOCKED_CSV = "data/test_locked_v14.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path):
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    miss, unexp = model.load_state_dict(ckpt, strict=False)
    assert len(miss) == 0 and len(unexp) == 0
    model.to(DEVICE)
    model.eval()
    return model

def get_dataloader(csv_file, batch_size=8, shuffle=False):
    audio_conf = {
        'num_mel_bins': 128, 'target_length': 1024,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'mode': 'eval', 'mean': -5.081, 'std': 4.4849,
        'noise': False, 'im_res': 224
    }
    dataset = dataloader.VideoAudioEvalDataset(csv_file=csv_file, audio_conf=audio_conf)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    return loader, dataset

def evaluate_model(model, loader, mode='normal'):
    """
    mode: 'normal', 'audio_only', 'video_only', 'zeros'
    """
    all_preds = []
    all_targets = []
    all_logits = []
    all_video_names = []

    model.eval()
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_names) in enumerate(loader):
            if mode == 'audio_only':
                v_input = torch.zeros_like(v_input)
            elif mode == 'video_only':
                a_input = torch.zeros_like(a_input)
            elif mode == 'zeros':
                a_input = torch.zeros_like(a_input)
                v_input = torch.zeros_like(v_input)

            a_input = a_input.to(DEVICE)
            v_input = v_input.to(DEVICE)

            with autocast():
                output = model(a_input, v_input)

            logits = output.cpu().numpy()
            for j in range(len(video_names)):
                true_label = labels[j][0].item()
                all_targets.append(int(true_label))
                all_logits.append(logits[j][0])
                all_video_names.append(video_names[j])

    return np.array(all_targets), np.array(all_logits), all_video_names

def run_experiment_A(targets, logits):
    print("\n" + "="*50)
    print("EXPERIMENT A - THRESHOLD SWEEP")
    print("="*50)
    probs = 1 / (1 + np.exp(-logits)) # Sigmoid
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    for th in thresholds:
        preds = (probs >= th).astype(int)
        acc = sk_metrics.accuracy_score(targets, preds)
        prec = sk_metrics.precision_score(targets, preds, zero_division=0)
        rec = sk_metrics.recall_score(targets, preds, zero_division=0)
        f1 = sk_metrics.f1_score(targets, preds, zero_division=0)
        bal_acc = sk_metrics.balanced_accuracy_score(targets, preds)
        
        tn, fp, fn, tp = sk_metrics.confusion_matrix(targets, preds).ravel()
        specificity = tn / (tn + fp)
        fpr = fp / (tn + fp)
        fnr = fn / (tp + fn)
        
        print(f"Threshold: {th:.2f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | BalAcc: {bal_acc:.4f} | Spec: {specificity:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f} | FP: {fp} | FN: {fn}")

def run_experiment_B(targets, logits):
    print("\n" + "="*50)
    print("EXPERIMENT B - CALIBRATION")
    print("="*50)
    
    # We will use 50% for fitting, 50% for eval to be clean.
    np.random.seed(42)
    indices = np.arange(len(targets))
    np.random.shuffle(indices)
    split = len(targets) // 2
    
    train_idx, test_idx = indices[:split], indices[split:]
    t_train, l_train = targets[train_idx], logits[train_idx]
    t_test, l_test = targets[test_idx], logits[test_idx]
    
    # 1. Uncalibrated
    probs_uncalib = 1 / (1 + np.exp(-l_test))
    brier_uncalib = sk_metrics.brier_score_loss(t_test, probs_uncalib)
    
    # 2. Platt Scaling (Logistic Regression on logits)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(l_train.reshape(-1, 1), t_train)
    probs_platt = lr.predict_proba(l_test.reshape(-1, 1))[:, 1]
    brier_platt = sk_metrics.brier_score_loss(t_test, probs_platt)
    
    # 3. Isotonic Regression
    iso = IsotonicRegression(out_of_bounds='clip')
    p_train = 1 / (1 + np.exp(-l_train))
    iso.fit(p_train, t_train)
    probs_iso = iso.predict(probs_uncalib)
    brier_iso = sk_metrics.brier_score_loss(t_test, probs_iso)
    
    print(f"Brier Uncalibrated: {brier_uncalib:.4f}")
    print(f"Brier Platt Scaling: {brier_platt:.4f}")
    print(f"Brier Isotonic: {brier_iso:.4f}")
    
    print("\nPlatt Scaling details:")
    print(f"  LR Coef: {lr.coef_[0][0]:.4f}, LR Intercept: {lr.intercept_[0]:.4f}")
    # Threshold at 0.5 for Platt
    preds_platt = (probs_platt >= 0.5).astype(int)
    tn, fp, fn, tp = sk_metrics.confusion_matrix(t_test, preds_platt).ravel()
    print(f"  Platt @ 0.5 -> FPR: {fp/(tn+fp):.4f}, FNR: {fn/(tp+fn):.4f}, F1: {sk_metrics.f1_score(t_test, preds_platt):.4f}")

def run_experiment_C(model, loader):
    print("\n" + "="*50)
    print("EXPERIMENT C - ZERO-INPUT BASELINE")
    print("="*50)
    targets, logits, _ = evaluate_model(model, loader, mode='zeros')
    probs = 1 / (1 + np.exp(-logits))
    print(f"Mean Fake Prob (Zeros): {np.mean(probs):.4f}")
    print(f"Std Fake Prob (Zeros): {np.std(probs):.4f}")
    print(f"Min Fake Prob: {np.min(probs):.4f}, Max Fake Prob: {np.max(probs):.4f}")
    # Should be deterministic, so std is ~0.
    
def run_experiment_D(loader, val_targets, val_logits):
    print("\n" + "="*50)
    print("EXPERIMENT D - CLASSIFIER-HEAD ANALYSIS")
    print("="*50)
    # Train the head only
    print("Setting up head-only training...")
    model = load_model(CHECKPOINT)
    
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze head
    for param in model.module.mlp_head.parameters():
        param.requires_grad = True
        
    train_loader, _ = get_dataloader(TRAIN_CSV, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.Adam(model.module.mlp_head.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    # 1 epoch is enough to see if it adjusts the boundary
    print("Training mlp_head for 1 epoch...")
    scaler = torch.cuda.amp.GradScaler()
    for i, (a_input, v_input, labels, _) in enumerate(train_loader):
        a_input = a_input.to(DEVICE)
        v_input = v_input.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            output = model(a_input, v_input)
            # Use index 0 for fake class to match previous logic
            loss = criterion(output[:, 0], labels[:, 0])
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (i+1) % 100 == 0:
            print(f"  Step {i+1}, Loss: {loss.item():.4f}")
            
    print("Evaluating tuned head on validation...")
    ft_targets, ft_logits, _ = evaluate_model(model, loader, mode='normal')
    
    probs = 1 / (1 + np.exp(-ft_logits))
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = sk_metrics.confusion_matrix(ft_targets, preds).ravel()
    f1 = sk_metrics.f1_score(ft_targets, preds)
    fpr = fp / (tn + fp)
    
    print(f"Fine-tuned Head @ 0.5 -> FPR: {fpr:.4f}, F1: {f1:.4f}, FP: {fp}, FN: {fn}")
    
def run_experiment_E(model, loader, targets, normal_logits):
    print("\n" + "="*50)
    print("EXPERIMENT E - MODALITY BALANCE")
    print("="*50)
    _, audio_logits, _ = evaluate_model(model, loader, mode='audio_only')
    _, video_logits, _ = evaluate_model(model, loader, mode='video_only')
    
    for name, log in [("Audio+Video", normal_logits), ("Audio-Only", audio_logits), ("Video-Only", video_logits)]:
        probs = 1 / (1 + np.exp(-log))
        preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = sk_metrics.confusion_matrix(targets, preds).ravel()
        f1 = sk_metrics.f1_score(targets, preds, zero_division=0)
        fpr = fp / (tn + fp)
        print(f"{name:<15} | FPR: {fpr:.4f} | F1: {f1:.4f} | FP: {fp:<4} | FN: {fn:<4}")

def run_experiment_F(targets, logits, vnames):
    print("\n" + "="*50)
    print("EXPERIMENT F - REAL-MEDIA REGRESSION")
    print("="*50)
    
    real_idx = np.where(targets == 0)[0]
    if len(real_idx) == 0:
        print("No REAL media found in this dataset.")
        return
        
    probs = 1 / (1 + np.exp(-logits))
    real_probs = probs[real_idx]
    
    print(f"Evaluated on {len(real_idx)} genuine videos.")
    print(f"Mean Fake Prob: {np.mean(real_probs):.4f}")
    print(f"Median: {np.median(real_probs):.4f}")
    print(f"P95: {np.percentile(real_probs, 95):.4f}")
    print(f"Max: {np.max(real_probs):.4f}")
    
    fp_50 = np.sum(real_probs >= 0.5)
    fp_60 = np.sum(real_probs >= 0.6)
    print(f"False Positives @ 0.5: {fp_50}")
    print(f"False Positives @ 0.6: {fp_60}")

def main():
    print("Loading V14 Validation Data...")
    val_loader, _ = get_dataloader(VAL_CSV, batch_size=16, shuffle=False)
    
    print("Loading Original V14 Model...")
    model = load_model(CHECKPOINT)
    
    print("Running Baseline Inference on Validation Set...")
    val_targets, val_logits, val_vnames = evaluate_model(model, val_loader, mode='normal')
    
    run_experiment_A(val_targets, val_logits)
    run_experiment_B(val_targets, val_logits)
    run_experiment_C(model, val_loader)
    run_experiment_E(model, val_loader, val_targets, val_logits)
    run_experiment_F(val_targets, val_logits, val_vnames)
    
    # Run D last since it mutates model weights
    run_experiment_D(val_loader, val_targets, val_logits)

if __name__ == "__main__":
    main()
