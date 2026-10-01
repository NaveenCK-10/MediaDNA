"""
V16 Phase 3 Evaluator: Visual Forensic Branch

Evaluates the new independent Visual Forensics Module on validation data
and paired demonstration videos.
"""

import os
import csv
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
from sklearn import metrics as sk_metrics

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT
from backend.modules.visual_forensics import VisualForensicsModule

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

def load_model(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def evaluate_visual_module(model, csv_file, is_paired=False):
    """
    Run evaluation using the VisualForensicsModule.
    """
    visual_module = VisualForensicsModule(model)
    
    results = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 50: # Limit for speed in this test run
                break
                
            if is_paired:
                video_path = os.path.join(PROJECT_ROOT, row['video_path'].replace('\\', '/'))
                ground_truth = int(row['label'])
                pair_id = row['pair_id']
            else:
                # validation csv (val_v14.csv) format: video_path,label,type
                video_path = os.path.join(PROJECT_ROOT, row['video_path'].replace('\\', '/'))
                ground_truth = int(row['label'])
                pair_id = None
            
            try:
                analysis = visual_module.analyze(video_path)
                
                results.append({
                    "video_path": row['video_path'],
                    "ground_truth": ground_truth,
                    "pair_id": pair_id,
                    "visual_anomaly_score": analysis["visual_anomaly_score"],
                    "faces_detected": analysis["faces_detected"],
                    "temporal_mae": analysis["metrics"]["temporal_mae"],
                    "spatial_variance": analysis["metrics"]["spatial_variance"]
                })
                
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1} videos...")
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                
    return results

def compute_metrics(results):
    targets = np.array([r["ground_truth"] for r in results])
    scores = np.array([r["visual_anomaly_score"] for r in results])
    
    # Evaluate as an anomaly score using ROC-AUC
    if len(np.unique(targets)) > 1:
        roc_auc = sk_metrics.roc_auc_score(targets, scores)
        pr_auc = sk_metrics.average_precision_score(targets, scores)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
        
    real_scores = scores[targets == 0]
    fake_scores = scores[targets == 1]
    
    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "mean_real_score": round(float(np.mean(real_scores)) if len(real_scores) > 0 else 0, 4),
        "mean_fake_score": round(float(np.mean(fake_scores)) if len(fake_scores) > 0 else 0, 4),
        "n_samples": len(results),
        "n_real": len(real_scores),
        "n_fake": len(fake_scores)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="v16_phase3_visual")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v14_fullscale/models/best_audio_model.pth")
    parser.add_argument("--val-csv", type=str, default="data/val_v14.csv")
    parser.add_argument("--demo-csv", type=str, default="data/paired_demo.csv")
    args = parser.parse_args()

    print("=" * 60)
    print(f"V16 PHASE 3 VISUAL EVALUATION: {args.exp_id}")
    print("=" * 60)
    
    model = load_model(args.checkpoint)
    
    print("\n1. Evaluating on Validation Subset...")
    val_results = evaluate_visual_module(model, os.path.join(PROJECT_ROOT, args.val_csv), is_paired=False)
    val_metrics = compute_metrics(val_results)
    
    print("\n2. Evaluating on Paired Demonstration Set...")
    demo_results = evaluate_visual_module(model, os.path.join(PROJECT_ROOT, args.demo_csv), is_paired=True)
    demo_metrics = compute_metrics(demo_results)
    
    # Save Report
    exp_dir = os.path.join(EXPERIMENTS_DIR, args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    report_path = os.path.join(exp_dir, "PHASE3_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# Phase 3 Execution Report: Visual Forensic Branch\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        f.write("## Validation Metrics (Subset)\n")
        f.write(f"- **ROC-AUC**: {val_metrics['roc_auc']}\n")
        f.write(f"- **PR-AUC**: {val_metrics['pr_auc']}\n")
        f.write(f"- **Mean Real Score**: {val_metrics['mean_real_score']}\n")
        f.write(f"- **Mean Fake Score**: {val_metrics['mean_fake_score']}\n\n")
        
        f.write("## Paired Demo Metrics\n")
        f.write(f"- **ROC-AUC**: {demo_metrics['roc_auc']}\n")
        f.write(f"- **Mean Real Score**: {demo_metrics['mean_real_score']}\n")
        f.write(f"- **Mean Fake Score**: {demo_metrics['mean_fake_score']}\n\n")
        
        f.write("## Paired Results Breakdown\n")
        f.write("| Pair ID | Real Score | Fake Score | Delta |\n")
        f.write("|---------|------------|------------|-------|\n")
        
        # Group pairs
        pairs = {}
        for r in demo_results:
            pid = r['pair_id']
            if pid not in pairs:
                pairs[pid] = {}
            if r['ground_truth'] == 0:
                pairs[pid]['real'] = r['visual_anomaly_score']
            else:
                pairs[pid]['fake'] = r['visual_anomaly_score']
                
        for pid, data in pairs.items():
            real_score = data.get('real', 0.0)
            fake_score = data.get('fake', 0.0)
            delta = fake_score - real_score
            f.write(f"| {pid} | {real_score:.4f} | {fake_score:.4f} | {delta:.4f} |\n")
            
    print(f"\nReport saved to: {report_path}")
    print(f"Validation AUC: {val_metrics['roc_auc']}")
    print(f"Paired Demo AUC: {demo_metrics['roc_auc']}")

if __name__ == "__main__":
    main()
