"""
Phase 11-12: Final held-out test evaluation for OpenAVFF medium-scale baseline.
Runs the trained model on the test set and produces per-category metrics.

Usage:
    python evaluate_medium.py --checkpoint exp\stage-3-medium\models\best_audio_model.pth
"""
import os
import sys
import csv
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn import metrics as sk_metrics
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="data/test_medium.csv")
    parser.add_argument("--annotated_csv", type=str, default="data/test_medium_annotated.csv")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("OpenAVFF Medium-Scale Test Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Device: {device}")

    # Load model
    model = VideoCAVMAEFT()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    miss, unexp = model.load_state_dict(ckpt, strict=False)
    print(f"Model loaded. Missing: {len(miss)}, Unexpected: {len(unexp)}")
    assert len(miss) == 0 and len(unexp) == 0, "Checkpoint mismatch!"

    model.to(device)
    model.eval()

    # Setup dataloader
    val_audio_conf = {
        'num_mel_bins': 128, 'target_length': 1024,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'mode': 'eval', 'mean': -5.081, 'std': 4.4849,
        'noise': False, 'im_res': 224
    }
    test_dataset = dataloader.VideoAudioEvalDataset(
        csv_file=args.test_csv, audio_conf=val_audio_conf
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    # Load category annotations
    categories = {}
    if os.path.exists(args.annotated_csv):
        with open(args.annotated_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    categories[row[0]] = row[2]
        print(f"Loaded {len(categories)} category annotations")

    # Run inference
    all_preds = []
    all_targets = []
    all_logits = []
    all_video_names = []
    all_cats = []

    print("\nRunning inference...")
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_names) in enumerate(test_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                output = model(a_input, v_input)

            logits = output.cpu().numpy()
            probs = torch.sigmoid(output).cpu().numpy()

            for j in range(len(video_names)):
                vname = video_names[j]
                true_label = labels[j][0].item()  # [label, 1-label], index 0 is fake-class
                pred_prob = probs[j][0]  # P(Fake)
                pred_label = 1 if pred_prob >= 0.5 else 0

                all_preds.append(pred_label)
                all_targets.append(int(true_label))
                all_logits.append(logits[j])
                all_video_names.append(vname)
                all_cats.append(categories.get(vname, 'Unknown'))

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(test_loader)} batches")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)

    # ========== OVERALL METRICS ==========
    print("\n" + "=" * 60)
    print("OVERALL TEST METRICS")
    print("=" * 60)

    accuracy = sk_metrics.accuracy_score(all_targets, all_preds)
    precision = sk_metrics.precision_score(all_targets, all_preds, zero_division=0)
    recall = sk_metrics.recall_score(all_targets, all_preds, zero_division=0)
    f1 = sk_metrics.f1_score(all_targets, all_preds, zero_division=0)

    # For mAP and AUC, use the raw logits (index 0 = Fake)
    fake_logits = all_logits[:, 0]
    try:
        auc = sk_metrics.roc_auc_score(all_targets, fake_logits)
    except ValueError:
        auc = -1
    try:
        mAP = sk_metrics.average_precision_score(all_targets, fake_logits)
    except ValueError:
        mAP = -1

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  mAP:       {mAP:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    # Confusion matrix
    cm = sk_metrics.confusion_matrix(all_targets, all_preds)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"  Actual Real:   {cm[0][0]:>10}    {cm[0][1]:>10}")
    print(f"  Actual Fake:   {cm[1][0]:>10}    {cm[1][1]:>10}")

    total_pred_real = np.sum(all_preds == 0)
    total_pred_fake = np.sum(all_preds == 1)
    print(f"\n  Total predicted Real: {total_pred_real}")
    print(f"  Total predicted Fake: {total_pred_fake}")

    # ========== PER-CATEGORY METRICS ==========
    print("\n" + "=" * 60)
    print("PER-CATEGORY RESULTS")
    print("=" * 60)

    cat_names = ['FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio', 'RealVideo-RealAudio']
    for cat in cat_names:
        mask = np.array([c == cat for c in all_cats])
        if mask.sum() == 0:
            print(f"\n  {cat}: No samples")
            continue

        cat_preds = all_preds[mask]
        cat_targets = all_targets[mask]
        cat_logits = all_logits[mask]

        cat_acc = sk_metrics.accuracy_score(cat_targets, cat_preds)
        expected_label = 0 if cat == 'RealVideo-RealAudio' else 1
        correct = np.sum(cat_preds == expected_label)

        print(f"\n  {cat} ({mask.sum()} samples, expected label={expected_label}):")
        print(f"    Accuracy (overall):   {cat_acc:.4f}")
        print(f"    Correct predictions:  {correct}/{mask.sum()}")

        # Show sample logits
        sample_indices = np.where(mask)[0][:5]
        for idx in sample_indices:
            logit_fake = all_logits[idx][0]
            logit_real = all_logits[idx][1]
            pred = all_preds[idx]
            print(f"    Sample: logit_fake={logit_fake:.3f}, logit_real={logit_real:.3f}, pred={pred}")

    # ========== WRITE RESULTS ==========
    report_path = "exp/stage-3-medium/test_results.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("OpenAVFF Medium-Scale Baseline — Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(all_targets)}\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"mAP:       {mAP:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                 Predicted Real  Predicted Fake\n")
        f.write(f"  Actual Real:   {cm[0][0]:>10}    {cm[0][1]:>10}\n")
        f.write(f"  Actual Fake:   {cm[1][0]:>10}    {cm[1][1]:>10}\n\n")

        for cat in cat_names:
            mask = np.array([c == cat for c in all_cats])
            if mask.sum() == 0:
                continue
            cat_preds_c = all_preds[mask]
            cat_targets_c = all_targets[mask]
            expected_label = 0 if cat == 'RealVideo-RealAudio' else 1
            correct = np.sum(cat_preds_c == expected_label)
            f.write(f"{cat} ({mask.sum()} samples): {correct}/{mask.sum()} correct\n")

    print(f"\nResults saved to: {report_path}")
    print("Evaluation complete.")

if __name__ == '__main__':
    main()
