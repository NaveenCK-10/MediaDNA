"""
OpenAVFF Comprehensive Diagnostic Script
Inspects every stage of the evaluation pipeline without modifying model or checkpoint.
"""
import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import argparse
import os
import glob
import csv
import random

def create_balanced_csv(fakeavceleb_root, output_csv, n_per_category=20):
    """Create a balanced diagnostic CSV with videos from all 4 categories."""
    categories = {
        'FakeVideo-FakeAudio': 1,
        'RealVideo-RealAudio': 0,
        'FakeVideo-RealAudio': 1,
        'RealVideo-FakeAudio': 1,
    }
    
    rows = []
    for cat_name, label in categories.items():
        cat_dir = os.path.join(fakeavceleb_root, cat_name)
        if not os.path.exists(cat_dir):
            print(f"WARNING: {cat_dir} does not exist, skipping")
            continue
        
        # Find all MP4 files
        mp4_files = []
        for root, dirs, files in os.walk(cat_dir):
            for f in files:
                if f.lower().endswith('.mp4'):
                    mp4_files.append(os.path.join(root, f))
        
        print(f"  {cat_name}: found {len(mp4_files)} MP4 files")
        
        if len(mp4_files) < n_per_category:
            print(f"  WARNING: only {len(mp4_files)} files available, using all")
            selected = mp4_files
        else:
            random.seed(42)
            selected = random.sample(mp4_files, n_per_category)
        
        for vpath in selected:
            rows.append((vpath, label, cat_name))
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label'])
        for vpath, label, _ in rows:
            writer.writerow([vpath, label])
    
    print(f"\nCreated {output_csv} with {len(rows)} samples")
    return rows


def diagnose(csv_file, checkpoint_path, create_balanced=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    
    # =========================================================
    # SECTION 1: Model + Checkpoint Loading Diagnosis
    # =========================================================
    print("\n" + "="*70)
    print("SECTION 1: MODEL & CHECKPOINT LOADING")
    print("="*70)
    
    audio_model = VideoCAVMAEFT()
    audio_model = torch.nn.DataParallel(audio_model)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint type: {type(ckpt)}")
    print(f"Checkpoint keys count: {len(ckpt)}")
    
    # Check for classifier head keys
    head_keys = [k for k in ckpt.keys() if 'mlp_head' in k or 'mlp_vision' in k or 'mlp_audio' in k]
    print(f"\nClassifier head keys in checkpoint:")
    for k in head_keys:
        print(f"  {k}: shape={ckpt[k].shape}, dtype={ckpt[k].dtype}")
    
    miss, unexp = audio_model.load_state_dict(ckpt, strict=False)
    print(f"\nMissing keys ({len(miss)}):")
    for k in miss:
        print(f"  {k}")
    print(f"Unexpected keys ({len(unexp)}):")
    for k in unexp:
        print(f"  {k}")
    
    # Print classifier head architecture
    print("\nMLP head architecture:")
    print(audio_model.module.mlp_head)
    print(f"\nmlp_head.fc3 output features: {audio_model.module.mlp_head.fc3.out_features}")
    print(f"mlp_head.fc3 weight shape: {audio_model.module.mlp_head.fc3.weight.shape}")
    print(f"mlp_head.fc3 bias: {audio_model.module.mlp_head.fc3.bias}")
    
    # =========================================================
    # SECTION 2: Eval mode diagnosis
    # =========================================================
    print("\n" + "="*70)
    print("SECTION 2: MODEL MODE CHECK")
    print("="*70)
    
    print(f"Model training mode BEFORE .eval(): {audio_model.training}")
    audio_model.eval()
    print(f"Model training mode AFTER .eval(): {audio_model.training}")
    
    # Check dropout layers
    for name, module in audio_model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"  Dropout layer: {name}, p={module.p}, training={module.training}")
    
    audio_model.to(device)
    
    # =========================================================
    # SECTION 3: Dataset & Preprocessing Diagnosis
    # =========================================================
    print("\n" + "="*70)
    print("SECTION 3: DATASET & PREPROCESSING")
    print("="*70)
    
    dataset_mean = -5.081
    dataset_std = 4.4849
    target_length = 1024
    val_audio_conf = {
        'num_mel_bins': 128,
        'target_length': target_length,
        'freqm': 0, 'timem': 0, 'mixup': 0,
        'mode': 'eval',
        'mean': dataset_mean, 'std': dataset_std,
        'noise': False, 'im_res': 224
    }
    
    dataset = dataloader.VideoAudioEvalDataset(csv_file=csv_file, audio_conf=val_audio_conf)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Normalization: mean={dataset_mean}, std={dataset_std}")
    print(f"Target length: {target_length}")
    
    # =========================================================
    # SECTION 4: Per-Sample Inference Diagnosis
    # =========================================================
    print("\n" + "="*70)
    print("SECTION 4: PER-SAMPLE INFERENCE")
    print("="*70)
    
    all_preds = []
    all_labels = []
    all_categories = []
    
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_names) in enumerate(val_loader):
            video_name = video_names[0]
            
            # Determine category from path
            if 'FakeVideo-FakeAudio' in video_name:
                category = 'FakeVideo-FakeAudio'
            elif 'FakeVideo-RealAudio' in video_name:
                category = 'FakeVideo-RealAudio'
            elif 'RealVideo-FakeAudio' in video_name:
                category = 'RealVideo-FakeAudio'
            elif 'RealVideo-RealAudio' in video_name:
                category = 'RealVideo-RealAudio'
            else:
                category = 'Unknown'
            
            label_tensor = labels.cpu().numpy()[0]
            csv_label = int(label_tensor[0])
            
            # Audio tensor stats (BEFORE model, AFTER normalization)
            a_stats = {
                'shape': tuple(a_input.shape),
                'min': a_input.min().item(),
                'max': a_input.max().item(),
                'mean': a_input.mean().item(),
                'std': a_input.std().item(),
            }
            
            # Visual tensor stats
            v_stats = {
                'shape': tuple(v_input.shape),
                'min': v_input.min().item(),
                'max': v_input.max().item(),
                'mean': v_input.mean().item(),
                'std': v_input.std().item(),
            }
            
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            
            with autocast():
                audio_output = audio_model(a_input, v_input)
            
            raw_logits = audio_output.cpu().float().numpy()[0]
            sigmoid_out = torch.sigmoid(audio_output.float()).cpu().numpy()[0]
            
            # Predicted class: argmax of raw logits
            # Index 0 = Fake (label=1), Index 1 = Real (label=0)
            pred_idx = np.argmax(raw_logits)
            pred_class = 1 if pred_idx == 0 else 0
            fake_score = sigmoid_out[0]
            
            all_preds.append(pred_class)
            all_labels.append(csv_label)
            all_categories.append(category)
            
            # Print first 10 or all if < 80
            if i < 10 or len(dataset) <= 80:
                print(f"\n--- Sample {i+1} ---")
                print(f"  VIDEO: {os.path.basename(video_name)}")
                print(f"  CATEGORY: {category}")
                print(f"  CSV LABEL: {csv_label}  (target tensor: {label_tensor})")
                print(f"  Audio tensor: shape={a_stats['shape']}, "
                      f"min={a_stats['min']:.4f}, max={a_stats['max']:.4f}, "
                      f"mean={a_stats['mean']:.4f}, std={a_stats['std']:.4f}")
                print(f"  Visual tensor: shape={v_stats['shape']}, "
                      f"min={v_stats['min']:.4f}, max={v_stats['max']:.4f}, "
                      f"mean={v_stats['mean']:.4f}, std={v_stats['std']:.4f}")
                print(f"  RAW LOGITS: {raw_logits}")
                print(f"  SIGMOID: {sigmoid_out}")
                print(f"  PREDICTED CLASS: {pred_class}")
                print(f"  FAKE SCORE: {fake_score:.6f}")
    
    # =========================================================
    # SECTION 5: Aggregate Metrics
    # =========================================================
    print("\n" + "="*70)
    print("SECTION 5: AGGREGATE METRICS")
    print("="*70)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_categories = np.array(all_categories)
    
    # Overall
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total
    
    # True Positives, False Positives, etc.
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:     Fake    Real")
    print(f"  Actual Fake:   {tp:4d}    {fn:4d}")
    print(f"  Actual Real:   {fp:4d}    {tn:4d}")
    
    # Per-category
    print(f"\nPer-Category Results:")
    for cat in sorted(set(all_categories)):
        mask = all_categories == cat
        cat_preds = all_preds[mask]
        cat_labels = all_labels[mask]
        cat_correct = (cat_preds == cat_labels).sum()
        cat_total = len(cat_labels)
        cat_acc = cat_correct / cat_total if cat_total > 0 else 0
        pred_fake = (cat_preds == 1).sum()
        pred_real = (cat_preds == 0).sum()
        print(f"  {cat}: acc={cat_acc:.4f} ({cat_correct}/{cat_total}), "
              f"predicted_fake={pred_fake}, predicted_real={pred_real}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage-3.pth")
    parser.add_argument("--csv_file", type=str, default=None)
    parser.add_argument("--create_balanced", action='store_true')
    args = parser.parse_args()
    
    fakeavceleb_root = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    balanced_csv = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\diagnostic_balanced.csv"
    
    if args.create_balanced:
        print("="*70)
        print("Creating balanced diagnostic CSV")
        print("="*70)
        create_balanced_csv(fakeavceleb_root, balanced_csv, n_per_category=20)
    
    csv_file = args.csv_file if args.csv_file else balanced_csv
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        print("Run with --create_balanced first")
        return
    
    diagnose(csv_file, args.checkpoint)


if __name__ == '__main__':
    main()
