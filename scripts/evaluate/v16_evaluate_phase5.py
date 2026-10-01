"""
V16 Phase 5 Evaluator: Temporal Forensic Branch

Evaluates the Temporal Forensics Module on validation data
and paired demonstration videos. Separates multimodal manipulation types.
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

from backend.modules.temporal_forensics import TemporalForensicsModule

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

def evaluate_temporal_module(csv_file, is_paired=False):
    """
    Run evaluation using the TemporalForensicsModule.
    """
    temporal_module = TemporalForensicsModule()
    
    results = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 50: # Limit for speed
                break
                
            video_path = os.path.join(PROJECT_ROOT, row['video_path'].replace('\\', '/'))
            ground_truth = int(row['label'])
            
            if is_paired:
                pair_id = row['pair_id']
                # Determine manipulation type based on the 'type' column
                # E.g., 'RealVideo-RealAudio', 'FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio'
                modality_type = row.get('type', 'Unknown')
            else:
                pair_id = None
                modality_type = row.get('type', 'Unknown')
            
            try:
                analysis = temporal_module.analyze(video_path)
                
                results.append({
                    "video_path": row['video_path'],
                    "ground_truth": ground_truth,
                    "pair_id": pair_id,
                    "modality_type": modality_type,
                    "temporal_anomaly_score": analysis["temporal_anomaly_score"],
                    "mean_shift": analysis["metrics"]["mean_shift"],
                    "max_shift": analysis["metrics"]["max_shift"],
                    "shift_variance": analysis["metrics"]["shift_variance"],
                    "temporal_evidence": analysis["temporal_evidence"]
                })
                
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1} videos...")
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                
    return results

def compute_metrics(results):
    targets = np.array([r["ground_truth"] for r in results])
    scores = np.array([r["temporal_anomaly_score"] for r in results])
    
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
    parser.add_argument("--exp-id", type=str, default="v16_phase5_temporal")
    parser.add_argument("--val-csv", type=str, default="data/val_v14.csv")
    parser.add_argument("--demo-csv", type=str, default="data/paired_demo.csv")
    args = parser.parse_args()

    print("=" * 60)
    print(f"V16 PHASE 5 TEMPORAL EVALUATION: {args.exp_id}")
    print("=" * 60)
    
    print("\n1. Evaluating on Validation Subset...")
    val_results = evaluate_temporal_module(os.path.join(PROJECT_ROOT, args.val_csv), is_paired=False)
    val_metrics = compute_metrics(val_results)
    
    print("\n2. Evaluating on Paired Demonstration Set...")
    demo_results = evaluate_temporal_module(os.path.join(PROJECT_ROOT, args.demo_csv), is_paired=True)
    demo_metrics = compute_metrics(demo_results)
    
    # Save Report
    exp_dir = os.path.join(EXPERIMENTS_DIR, args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    report_path = os.path.join(exp_dir, "PHASE5_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# Phase 5 Execution Report: Temporal Forensic Branch\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        
        f.write("## 1. Objective\n")
        f.write("Evaluate the independent temporal pathway, extracting frame-to-frame feature discontinuities (flickering/blending artifacts) independently of the main OpenAVFF classifier head.\n\n")
        
        f.write("## 2. Model & Features\n")
        f.write("- **Model**: Frozen MobileNetV2 (Frame Encoder)\n")
        f.write("- **Temporal Signals**: Frame-to-frame L2 Distance, Max Shift, Shift Variance\n\n")
        
        f.write("## 3. Validation Metrics\n")
        f.write(f"- **ROC-AUC**: {val_metrics['roc_auc']}\n")
        f.write(f"- **PR-AUC**: {val_metrics['pr_auc']}\n")
        f.write(f"- **Mean Real Score**: {val_metrics['mean_real_score']}\n")
        f.write(f"- **Mean Fake Score**: {val_metrics['mean_fake_score']}\n\n")
        
        f.write("## 4. Paired Demo Metrics\n")
        f.write(f"- **ROC-AUC**: {demo_metrics['roc_auc']}\n")
        f.write(f"- **Mean Real Score**: {demo_metrics['mean_real_score']}\n")
        f.write(f"- **Mean Fake Score**: {demo_metrics['mean_fake_score']}\n\n")
        
        f.write("## 5. Manipulation Type Analysis (Demo Set)\n")
        
        types = {}
        for r in demo_results:
            t = r['modality_type']
            if t not in types:
                types[t] = []
            types[t].append(r['temporal_anomaly_score'])
            
        for t, scores in types.items():
            f.write(f"- **{t}**: Mean Anomaly Score = {np.mean(scores):.4f} (n={len(scores)})\n")
            
        f.write("\n## 6. Sample Temporal Evidence (Localization)\n")
        # Print top 3 anomalies
        top_anomalies = sorted(demo_results, key=lambda x: x['temporal_anomaly_score'], reverse=True)[:3]
        for r in top_anomalies:
            f.write(f"- **Pair**: {r['pair_id']} ({r['modality_type']})\n")
            if r['temporal_evidence']:
                ev = r['temporal_evidence'][0]
                f.write(f"  - Peak Shift at: {ev['interval']} (Frames {ev['frame_indices']})\n")
                f.write(f"  - Shift Magnitude: {ev['shift_magnitude']:.2f}\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Validation AUC: {val_metrics['roc_auc']}")
    print(f"Paired Demo AUC: {demo_metrics['roc_auc']}")

if __name__ == "__main__":
    main()
