import os
import subprocess
import sys
import datetime
import torch

def run_experiment(exp_name, lr="1e-5", head_lr="50"):
    print("="*60)
    print(f"Starting V8 Experiment: {exp_name}")
    print("="*60)
    
    TRAIN_CSV = "data/train_v8.csv"
    VAL_CSV = "data/val_v8.csv"
    INITIAL_CHECKPOINT = r"exp\stage-3-local\models\best_audio_model.pth"
    SAVE_DIR = os.path.join(r"checkpoints\v8", exp_name)
    
    if not os.path.exists(INITIAL_CHECKPOINT):
        print(f"Error: Initial checkpoint not found at {INITIAL_CHECKPOINT}")
        return
        
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'models'), exist_ok=True)
    
    # Save config
    with open(os.path.join(SAVE_DIR, "config.txt"), 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Base LR: {lr}\n")
        f.write(f"Head LR Mult: {head_lr}\n")
        f.write(f"Dataset: train_v8.csv (800), val_v8.csv (200)\n")
        
    cmd = [
        sys.executable, "-W", "ignore", r"src\run_ft.py",
        "--data-train", TRAIN_CSV,
        "--data-val", VAL_CSV,
        "--save-dir", SAVE_DIR,
        "--n_classes", "2",
        "--lr", lr,
        "--head_lr", head_lr,
        "--n-epochs", "5",
        "--batch-size", "1",
        "--num_workers", "0",
        "--lrscheduler_start", "2",
        "--lrscheduler_decay", "0.5",
        "--lrscheduler_step", "1",
        "--dataset_mean", "-5.081",
        "--dataset_std", "4.4849",
        "--target_length", "1024",
        "--noise", "True",
        "--norm_pix_loss", "True",
        "--mae_loss_weight", "1.0",
        "--contrast_loss_weight", "0.01",
        "--loss", "BCE",
        "--metrics", "mAP",
        "--warmup", "True",
        "--pretrain_path", INITIAL_CHECKPOINT
    ]
    
    print("Executing command:")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return
        
    print(f"\nExperiment {exp_name} completed. Saved to {SAVE_DIR}")
    
if __name__ == '__main__':
    # 1. Baseline (like Stage 3, Head LR = 50)
    run_experiment("baseline", lr="1e-5", head_lr="50")
    
    # 2. Full Fine-Tuning (Head LR = 1, slower but unfreezes everything evenly)
    # run_experiment("full_finetune", lr="1e-5", head_lr="1")
