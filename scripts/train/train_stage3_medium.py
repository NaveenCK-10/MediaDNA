"""
OpenAVFF Medium-Scale Stage-3 Training Script
Launches the repository's actual training pipeline with local single-GPU settings.
Records initial/final weight hashes for verification.
"""
import os
import sys
import subprocess
import torch
import hashlib
import json
import datetime

# === Configuration ===
SEED = 42
TRAIN_CSV = r"data\train_medium.csv"
VAL_CSV = r"data\val_medium.csv"
INIT_CHECKPOINT = r"exp\stage-3-local\models\best_audio_model.pth"
SAVE_DIR = r"exp\stage-3-medium"
CONFIG_FILE = os.path.join(SAVE_DIR, "config.txt")

# Training parameters (from repository egs/stage-3.sh)
PARAMS = {
    'lr': '1e-5',
    'head_lr': '50',
    'n-epochs': '10',
    'batch-size': '1',
    'num_workers': '0',
    'lrscheduler_start': '2',
    'lrscheduler_decay': '0.5',
    'lrscheduler_step': '1',
    'dataset_mean': '-5.081',
    'dataset_std': '4.4849',
    'target_length': '1024',
    'noise': 'True',
    'norm_pix_loss': 'True',
    'mae_loss_weight': '1.0',
    'contrast_loss_weight': '0.01',
    'loss': 'BCE',
    'metrics': 'mAP',
    'warmup': 'True',
    'n_classes': '2',
    'n_print_steps': '100',
}

def get_weight_stats(ckpt_path):
    """Extract classifier head weight statistics."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    stats = {}
    keys = [
        'module.mlp_head.fc1.weight', 'module.mlp_head.fc1.bias',
        'module.mlp_head.fc2.weight', 'module.mlp_head.fc2.bias',
        'module.mlp_head.fc3.weight', 'module.mlp_head.fc3.bias',
    ]
    for key in keys:
        if key in ckpt:
            t = ckpt[key].float()
            stats[key] = {
                'mean': f"{t.mean().item():.6f}",
                'std': f"{t.std().item():.6f}",
                'hash': hashlib.md5(t.numpy().tobytes()).hexdigest()[:8],
            }
    return stats

def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    print("=" * 60)
    print("OpenAVFF Medium-Scale Stage-3 Training")
    print("=" * 60)
    print(f"Start time: {datetime.datetime.now()}")
    print(f"Init checkpoint: {INIT_CHECKPOINT}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV: {VAL_CSV}")
    print(f"Save dir: {SAVE_DIR}")
    
    # Verify init checkpoint exists
    if not os.path.exists(INIT_CHECKPOINT):
        print(f"ERROR: Init checkpoint not found: {INIT_CHECKPOINT}")
        sys.exit(1)
    
    # Create experiment directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'models'), exist_ok=True)
    
    # Record configuration
    with open(CONFIG_FILE, 'w') as f:
        f.write(f"Experiment: OpenAVFF Medium-Scale Stage-3 Baseline\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Init checkpoint: {INIT_CHECKPOINT}\n")
        f.write(f"Train CSV: {TRAIN_CSV}\n")
        f.write(f"Val CSV: {VAL_CSV}\n")
        f.write(f"GPU: NVIDIA GeForce RTX 4070 Laptop GPU\n")
        f.write(f"VRAM: 8 GB\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA: {torch.cuda.is_available()}\n")
        f.write(f"AMP: autocast + GradScaler (repository built-in)\n\n")
        f.write("Training Parameters:\n")
        for k, v in PARAMS.items():
            f.write(f"  {k}: {v}\n")
    print(f"Config saved to: {CONFIG_FILE}")
    
    # Record initial weights
    print("\nRecording initial classifier weights...")
    init_stats = get_weight_stats(INIT_CHECKPOINT)
    for k, v in init_stats.items():
        print(f"  {k}: hash={v['hash']}, mean={v['mean']}, std={v['std']}")
    
    # Build training command
    cmd = [
        sys.executable, "-W", "ignore", r"src\run_ft.py",
        "--data-train", TRAIN_CSV,
        "--data-val", VAL_CSV,
        "--save-dir", SAVE_DIR,
        "--pretrain_path", INIT_CHECKPOINT,
    ]
    for k, v in PARAMS.items():
        cmd.extend([f"--{k}", v])
    
    print(f"\nTraining command:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    # Execute training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining FAILED with exit code {e.returncode}")
        sys.exit(1)
    
    # Verify output checkpoint exists
    best_ckpt = os.path.join(SAVE_DIR, 'models', 'best_audio_model.pth')
    if not os.path.exists(best_ckpt):
        print(f"\nERROR: Best checkpoint not found: {best_ckpt}")
        sys.exit(1)
    
    # Record final weights
    print("\n" + "=" * 60)
    print("Weight Update Verification")
    print("=" * 60)
    final_stats = get_weight_stats(best_ckpt)
    
    any_changed = False
    for k in init_stats:
        if k in final_stats:
            i_hash = init_stats[k]['hash']
            f_hash = final_stats[k]['hash']
            changed = i_hash != f_hash
            status = "CHANGED" if changed else "UNCHANGED"
            print(f"  {k}: {i_hash} -> {f_hash} [{status}]")
            if changed:
                any_changed = True
    
    if any_changed:
        print("\n✓ Classifier weights successfully updated during training.")
    else:
        print("\n✗ WARNING: No classifier weights changed!")
    
    # Parse result.csv
    result_csv = os.path.join(SAVE_DIR, 'result.csv')
    if os.path.exists(result_csv):
        import numpy as np
        results = np.loadtxt(result_csv, delimiter=',')
        print(f"\n{'Epoch':<8} {'Acc':<10} {'mAP':<10} {'AUC':<10} {'LR':<12}")
        for i, row in enumerate(results):
            if row[0] != 0:  # skip empty rows
                print(f"{i+1:<8} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<12.2e}")
        
        best_idx = np.argmax(results[:, 1])  # best mAP
        print(f"\nBest epoch by mAP: {best_idx + 1} (mAP={results[best_idx, 1]:.4f}, Acc={results[best_idx, 0]:.4f})")
    
    print(f"\nBest checkpoint saved at: {best_ckpt}")
    print(f"Finished at: {datetime.datetime.now()}")

if __name__ == '__main__':
    main()
