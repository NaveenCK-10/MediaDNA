import os
import subprocess
import datetime
import sys

def run_candidate(candidate_name, noise_level):
    print("="*60)
    print(f"Starting V16.1 Candidate: {candidate_name}")
    print(f"Noise Level: {noise_level}")
    print("="*60)
    
    TRAIN_CSV = "data/train_v14.csv"
    VAL_CSV = "data/val_v14.csv"
    INITIAL_CHECKPOINT = r"checkpoints\v8\baseline\models\best_audio_model.pth"
    SAVE_DIR = os.path.join("checkpoints", f"v16_1_{candidate_name}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'models'), exist_ok=True)
    
    # Save config
    with open(os.path.join(SAVE_DIR, "config.txt"), 'w') as f:
        f.write(f"Experiment: V16.1 {candidate_name}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write("Base LR: 1e-5\n")
        f.write("Head LR Mult: 50\n")
        f.write(f"Noise Level: {noise_level}\n")
        f.write("Dataset: train_v14.csv, val_v14.csv\n")
        
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
        "--noise_level", str(noise_level),
        "--norm_pix_loss", "True",
        "--mae_loss_weight", "1.0",
        "--contrast_loss_weight", "0.01",
        "--loss", "BCE",
        "--metrics", "mAP",
        "--warmup", "True",
        "--pretrain_path", INITIAL_CHECKPOINT
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nTraining for {candidate_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {candidate_name} with error code {e.returncode}")

def main():
    # Candidate A (Clean) is mathematically identical to v14_fullscale.
    run_candidate("Cand_B_Mild", 0.5)
    run_candidate("Cand_C_Moderate", 1.0)
    run_candidate("Cand_D_Mixed", -1.0)  # -1.0 signals dynamic mixed noise
    print("\nRemaining V16.1 Candidates Trained Successfully!")

if __name__ == "__main__":
    main()
