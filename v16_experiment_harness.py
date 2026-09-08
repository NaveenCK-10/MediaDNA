import os
import csv
import json
import torch
import numpy as np
from datetime import datetime
from sklearn import metrics as sk_metrics
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
from torch.cuda.amp import autocast
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
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
    return loader

def evaluate(model, loader, mode, noise_level=0.0):
    all_targets = []
    all_logits = []
    
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
                
            if noise_level > 0.0:
                # Add gaussian noise to audio
                noise = torch.randn_like(a_input) * noise_level
                a_input = a_input + noise

            a_input = a_input.to(DEVICE)
            v_input = v_input.to(DEVICE)

            with autocast():
                output = model(a_input, v_input)

            logits = output.cpu().numpy()
            for j in range(len(labels)):
                true_label = labels[j][0].item()
                all_targets.append(int(true_label))
                all_logits.append(logits[j][0])
                
            if (i+1) % 10 == 0:
                print(f"Processed batch {i+1} / {len(loader)}...")

    return np.array(all_targets), np.array(all_logits)

def compute_metrics(targets, probs, threshold=0.60):
    preds = (probs >= threshold).astype(int)
    
    acc = sk_metrics.accuracy_score(targets, preds)
    bal_acc = sk_metrics.balanced_accuracy_score(targets, preds)
    prec = sk_metrics.precision_score(targets, preds, zero_division=0)
    rec = sk_metrics.recall_score(targets, preds, zero_division=0)
    f1 = sk_metrics.f1_score(targets, preds, zero_division=0)
    
    tn, fp, fn, tp = sk_metrics.confusion_matrix(targets, preds).ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    # Handle single-class edge cases gracefully for AUC
    if len(np.unique(targets)) > 1:
        roc_auc = sk_metrics.roc_auc_score(targets, probs)
        pr_auc = sk_metrics.average_precision_score(targets, probs)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
    
    # Score distributions
    real_idx = np.where(targets == 0)[0]
    fake_idx = np.where(targets == 1)[0]
    
    real_mean = np.mean(probs[real_idx]) if len(real_idx) > 0 else 0
    fake_mean = np.mean(probs[fake_idx]) if len(fake_idx) > 0 else 0

    return {
        "threshold": threshold,
        "acc": acc, "bal_acc": bal_acc,
        "prec": prec, "rec": rec, "f1": f1,
        "fpr": fpr, "fnr": fnr,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "real_mean": real_mean, "fake_mean": fake_mean,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def save_experiment_results(exp_name, config, metrics_list):
    os.makedirs("v16_experiments", exist_ok=True)
    exp_dir = os.path.join("v16_experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    # Save detailed report
    report_path = os.path.join(exp_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# Experiment: {exp_name}\n\n")
        f.write("## Configuration\n```json\n")
        f.write(json.dumps(config, indent=4))
        f.write("\n```\n\n## Metrics\n")
        
        for m in metrics_list:
            th = m['threshold']
            f.write(f"### Threshold {th:.2f}\n")
            f.write(f"- **ROC-AUC**: {m['roc_auc']:.4f}\n")
            f.write(f"- **PR-AUC (mAP)**: {m['pr_auc']:.4f}\n")
            f.write(f"- **Accuracy**: {m['acc']:.4f}\n")
            f.write(f"- **Balanced Accuracy**: {m['bal_acc']:.4f}\n")
            f.write(f"- **F1**: {m['f1']:.4f}\n")
            f.write(f"- **Precision**: {m['prec']:.4f}\n")
            f.write(f"- **FPR**: {m['fpr']:.4f} ({m['fp']}/{m['fp']+m['tn']})\n")
            f.write(f"- **FNR**: {m['fnr']:.4f} ({m['fn']}/{m['fn']+m['tp']})\n")
            f.write(f"- **Real Mean Score**: {m['real_mean']:.4f}\n")
            f.write(f"- **Fake Mean Score**: {m['fake_mean']:.4f}\n\n")
            
    # Append to master CSV
    csv_path = "v16_experiments/V16_EXPERIMENT_RESULTS.csv"
    file_exists = os.path.exists(csv_path)
    
    # We'll save the primary threshold (0.60) to the CSV
    primary = next((m for m in metrics_list if m['threshold'] == 0.60), metrics_list[0])
    
    with open(csv_path, "a", newline="") as f:
        fieldnames = ["timestamp", "exp_name", "checkpoint", "dataset", "mode", "noise_level", 
                      "threshold", "roc_auc", "pr_auc", "acc", "bal_acc", "f1", "prec", "fpr", "fnr", "real_mean", "fake_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "exp_name": exp_name,
            "checkpoint": config['checkpoint'],
            "dataset": config['csv'],
            "mode": config['mode'],
            "noise_level": config['noise_level'],
            "threshold": primary['threshold'],
            "roc_auc": primary['roc_auc'],
            "pr_auc": primary['pr_auc'],
            "acc": primary['acc'],
            "bal_acc": primary['bal_acc'],
            "f1": primary['f1'],
            "prec": primary['prec'],
            "fpr": primary['fpr'],
            "fnr": primary['fnr'],
            "real_mean": primary['real_mean'],
            "fake_mean": primary['fake_mean']
        })
    print(f"\nExperiment {exp_name} saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="V16 Experiment Harness")
    parser.add_argument("--exp_name", type=str, required=True, help="Unique name for the experiment")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v14_fullscale/models/best_audio_model.pth")
    parser.add_argument("--csv", type=str, default="data/val_v14.csv")
    parser.add_argument("--mode", type=str, choices=["normal", "audio_only", "video_only", "zeros"], default="normal")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Std dev of Gaussian noise added to audio")
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"RUNNING V16 EXPERIMENT: {args.exp_name}")
    print("="*60)
    
    # 1. Load Dataloader
    loader = get_dataloader(args.csv, batch_size=args.batch_size)
    
    # 2. Load Model
    model = load_model(args.checkpoint)
    
    # 3. Evaluate
    print(f"\nStarting evaluation... (Mode: {args.mode}, Noise: {args.noise_level})")
    targets, logits = evaluate(model, loader, mode=args.mode, noise_level=args.noise_level)
    
    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    # 4. Sweep Thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65]
    metrics_list = []
    
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    
    for th in thresholds:
        m = compute_metrics(targets, probs, threshold=th)
        metrics_list.append(m)
        print(f"Thresh {th:.2f} | AUC: {m['roc_auc']:.4f} | Acc: {m['acc']:.4f} | BalAcc: {m['bal_acc']:.4f} | FPR: {m['fpr']:.4f} | FNR: {m['fnr']:.4f} | CM: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
        
    # 5. Save Results
    config = vars(args)
    save_experiment_results(args.exp_name, config, metrics_list)

if __name__ == "__main__":
    main()
