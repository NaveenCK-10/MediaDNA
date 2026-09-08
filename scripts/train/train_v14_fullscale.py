import os
import subprocess
import sys
import datetime

def run_v14_fullscale():
    print("="*60)
    print("Starting V14 Full-Scale Scientific Training")
    print("="*60)
    
    TRAIN_CSV = "data/train_v14.csv"
    VAL_CSV = "data/val_v14.csv"
    INITIAL_CHECKPOINT = r"checkpoints\v8\baseline\models\best_audio_model.pth"
    SAVE_DIR = r"checkpoints\v14_fullscale"
    
    if not os.path.exists(INITIAL_CHECKPOINT):
        print(f"Error: Initial checkpoint not found at {INITIAL_CHECKPOINT}")
        return
        
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'models'), exist_ok=True)
    
    # Save config
    with open(os.path.join(SAVE_DIR, "config.txt"), 'w') as f:
        f.write("Experiment: V14 Full-Scale\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write("Base LR: 1e-5\n")
        f.write("Head LR Mult: 50\n")
        f.write("Dataset: train_v14.csv, val_v14.csv (Zero Identity Leakage verified)\n")
        
    cmd = [
        sys.executable, "-W", "ignore", r"src\run_ft.py",
        "--data-train", TRAIN_CSV,
        "--data-val", VAL_CSV,
        "--save-dir", SAVE_DIR,
        "--n_classes", "2",
        "--lr", "1e-5",
        "--head_lr", "50",
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
        
    print(f"\nV14 Full-Scale Training completed. Saved to {SAVE_DIR}")
    
if __name__ == '__main__':
    run_v14_fullscale()
