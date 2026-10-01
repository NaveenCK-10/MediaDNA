"""
V16 Per-Sample Experiment Logger.

Records individual sample-level predictions for every evaluation run.
This enables failure analysis, paired comparison, and per-sample tracking
across experiments.

Output format (CSV per experiment):
    sample_id, video_path, ground_truth, prediction, fake_probability,
    raw_logit_0, raw_logit_1, experiment_id, checkpoint, mode, noise_level
"""

import os
import sys
import csv
import json
import time
import torch
import numpy as np
from datetime import datetime
from sklearn import metrics as sk_metrics
from torch.cuda.amp import autocast
import argparse

# Add project root to sys.path so we can import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")


def load_model(checkpoint_path):
    """Load and prepare model for evaluation."""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    miss, unexp = model.load_state_dict(ckpt, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (
        f"Checkpoint mismatch! Missing: {len(miss)}, Unexpected: {len(unexp)}"
    )
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded. Device: {DEVICE}, Eval mode: {not model.training}")
    return model


def get_dataloader(csv_file, batch_size=8):
    """Create evaluation dataloader with frozen V15.4 preprocessing."""
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'mode': 'eval',
        'mean': -5.081,
        'std': 4.4849,
        'noise': False,
        'im_res': 224
    }
    dataset = dataloader.VideoAudioEvalDataset(csv_file=csv_file, audio_conf=audio_conf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return loader


def evaluate_with_logging(model, loader, mode, noise_level=0.0):
    """
    Run evaluation and return per-sample results.
    
    Returns:
        list of dicts, one per sample:
        {
            "video_path": str,
            "ground_truth": int,  # 0=real, 1=fake
            "fake_probability": float,
            "raw_logit_0": float,
            "raw_logit_1": float,
            "prediction": int,  # at threshold 0.60
        }
    """
    sample_results = []

    model.eval()
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_names) in enumerate(loader):
            # Apply modality ablation
            if mode == 'audio_only':
                v_input = torch.zeros_like(v_input)
            elif mode == 'video_only':
                a_input = torch.zeros_like(a_input)
            elif mode == 'zeros':
                a_input = torch.zeros_like(a_input)
                v_input = torch.zeros_like(v_input)

            # Apply noise injection
            if noise_level > 0.0:
                noise = torch.randn_like(a_input) * noise_level
                a_input = a_input + noise

            a_input = a_input.to(DEVICE)
            v_input = v_input.to(DEVICE)

            with autocast():
                output = model(a_input, v_input)

            logits = output.cpu().float().numpy()
            probs = 1 / (1 + np.exp(-logits))  # sigmoid

            for j in range(len(labels)):
                true_label = int(labels[j][0].item())  # labels[j] = [fake_prob, real_prob]
                fake_prob = float(probs[j][0])
                pred = 1 if fake_prob >= 0.60 else 0

                sample_results.append({
                    "video_path": video_names[j] if isinstance(video_names[j], str) else video_names[j],
                    "ground_truth": true_label,
                    "fake_probability": round(fake_prob, 6),
                    "raw_logit_0": round(float(logits[j][0]), 6),
                    "raw_logit_1": round(float(logits[j][1]), 6),
                    "prediction": pred,
                })

            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}/{len(loader)} ({len(sample_results)} samples)")

    return sample_results


def compute_metrics(sample_results, threshold=0.60):
    """Compute aggregate metrics from per-sample results."""
    targets = np.array([s["ground_truth"] for s in sample_results])
    probs = np.array([s["fake_probability"] for s in sample_results])
    preds = (probs >= threshold).astype(int)

    acc = sk_metrics.accuracy_score(targets, preds)
    bal_acc = sk_metrics.balanced_accuracy_score(targets, preds)
    prec = sk_metrics.precision_score(targets, preds, zero_division=0)
    rec = sk_metrics.recall_score(targets, preds, zero_division=0)
    f1 = sk_metrics.f1_score(targets, preds, zero_division=0)

    cm = sk_metrics.confusion_matrix(targets, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

    if len(np.unique(targets)) > 1:
        roc_auc = sk_metrics.roc_auc_score(targets, probs)
        pr_auc = sk_metrics.average_precision_score(targets, probs)
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    real_idx = np.where(targets == 0)[0]
    fake_idx = np.where(targets == 1)[0]
    real_mean = float(np.mean(probs[real_idx])) if len(real_idx) > 0 else 0
    fake_mean = float(np.mean(probs[fake_idx])) if len(fake_idx) > 0 else 0

    return {
        "threshold": threshold,
        "n_samples": len(targets),
        "n_real": int(len(real_idx)),
        "n_fake": int(len(fake_idx)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "acc": round(acc, 4),
        "bal_acc": round(bal_acc, 4),
        "prec": round(prec, 4),
        "rec": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "real_mean_score": round(real_mean, 4),
        "fake_mean_score": round(fake_mean, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_experiment(exp_id, config, sample_results, metrics_by_threshold):
    """Save complete experiment results to experiments/ directory."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. Save per-sample results
    sample_csv = os.path.join(exp_dir, "samples.csv")
    fieldnames = ["video_path", "ground_truth", "prediction", "fake_probability",
                   "raw_logit_0", "raw_logit_1"]
    with open(sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sample_results:
            writer.writerow({k: s[k] for k in fieldnames})

    # 3. Save aggregate metrics
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics_by_threshold, f, indent=2)

    # 4. Save human-readable report
    with open(os.path.join(exp_dir, "report.md"), "w") as f:
        f.write(f"# Experiment: {exp_id}\n\n")
        f.write(f"**Timestamp:** {config['timestamp']}\n\n")
        f.write("## Configuration\n```json\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n```\n\n")

        for th_key, m in metrics_by_threshold.items():
            f.write(f"## Metrics @ Threshold {m['threshold']}\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            for k, v in m.items():
                if k != "threshold":
                    f.write(f"| {k} | {v} |\n")
            f.write("\n")

        # Failure analysis
        failures = [s for s in sample_results if s["ground_truth"] != s["prediction"]]
        f.write(f"## Failures ({len(failures)}/{len(sample_results)})\n\n")
        if failures:
            f.write("| Video | Truth | Pred | P(Fake) |\n|-------|-------|------|---------|\n")
            for s in failures[:50]:  # Cap at 50
                truth_str = "FAKE" if s["ground_truth"] == 1 else "REAL"
                pred_str = "FAKE" if s["prediction"] == 1 else "REAL"
                f.write(f"| {os.path.basename(s['video_path'])} | {truth_str} | {pred_str} | {s['fake_probability']:.4f} |\n")

    print(f"\nExperiment saved to: {exp_dir}")
    print(f"  - config.json")
    print(f"  - samples.csv ({len(sample_results)} rows)")
    print(f"  - metrics.json")
    print(f"  - report.md")

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="V16 Per-Sample Experiment Runner")
    parser.add_argument("--exp-id", type=str, required=True,
                        help="Unique experiment identifier (e.g. v16_baseline_normal)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/v14_fullscale/models/best_audio_model.pth")
    parser.add_argument("--csv", type=str, default="data/val_v14.csv")
    parser.add_argument("--mode", type=str,
                        choices=["normal", "audio_only", "video_only", "zeros"],
                        default="normal")
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print(f"V16 EXPERIMENT: {args.exp_id}")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Dataset:     {args.csv}")
    print(f"  Mode:        {args.mode}")
    print(f"  Noise:       {args.noise_level}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Device:      {DEVICE}")
    print("=" * 60)

    config = {
        "exp_id": args.exp_id,
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "mode": args.mode,
        "noise_level": args.noise_level,
        "batch_size": args.batch_size,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
        "preprocessing": {
            "num_frames": 16,
            "im_res": 224,
            "num_mel_bins": 128,
            "target_length": 1024,
            "audio_sample_rate": 16000,
            "dataset_mean": -5.081,
            "dataset_std": 4.4849,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "decision_threshold": 0.60,
        }
    }

    # 1. Load model
    model = load_model(args.checkpoint)

    # 2. Load data
    loader = get_dataloader(args.csv, batch_size=args.batch_size)

    # 3. Evaluate with per-sample logging
    print(f"\nRunning evaluation (mode={args.mode}, noise={args.noise_level})...")
    t0 = time.time()
    sample_results = evaluate_with_logging(model, loader, args.mode, args.noise_level)
    eval_time = time.time() - t0
    config["eval_time_seconds"] = round(eval_time, 2)

    # 4. Compute metrics at multiple thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    metrics_by_threshold = {}
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS -- {args.exp_id} ({len(sample_results)} samples, {eval_time:.1f}s)")
    print(f"{'=' * 80}")
    
    for th in thresholds:
        m = compute_metrics(sample_results, threshold=th)
        metrics_by_threshold[f"threshold_{th:.2f}"] = m
        
        marker = " << PRODUCTION" if th == 0.60 else ""
        print(f"  T={th:.2f} | AUC={m['roc_auc']:.4f} | Acc={m['acc']:.4f} | "
              f"BalAcc={m['bal_acc']:.4f} | F1={m['f1']:.4f} | "
              f"FPR={m['fpr']:.4f} | FNR={m['fnr']:.4f} | "
              f"CM=[TN={m['tn']},FP={m['fp']},FN={m['fn']},TP={m['tp']}]{marker}")

    # 5. Save everything
    save_experiment(args.exp_id, config, sample_results, metrics_by_threshold)

    # 6. Summary
    primary = metrics_by_threshold["threshold_0.60"]
    print(f"\n{'=' * 60}")
    print(f"PRIMARY RESULT (threshold=0.60)")
    print(f"  ROC-AUC:           {primary['roc_auc']}")
    print(f"  Balanced Accuracy: {primary['bal_acc']}")
    print(f"  FPR:               {primary['fpr']}")
    print(f"  FNR:               {primary['fnr']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
