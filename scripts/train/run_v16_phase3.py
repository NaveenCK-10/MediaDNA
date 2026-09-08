import os
import subprocess

def run_evaluation(cand_name, ckpt_path, noise_name, noise_level):
    exp_name = f"Phase3_{cand_name}_{noise_name}"
    print(f"Running {exp_name} with noise level {noise_level}...")
    cmd = [
        "python", "v16_experiment_harness.py",
        "--exp_name", exp_name,
        "--mode", "normal",
        "--noise_level", str(noise_level),
        "--csv", "data/val_v14.csv",
        "--checkpoint", ckpt_path
    ]
    subprocess.run(cmd, check=True)

def main():
    candidates = {
        "Cand_A": "checkpoints/v14_fullscale/models/best_audio_model.pth",  # Clean is mathematically identical to V14 baseline
        "Cand_B": "checkpoints/v16_1_Cand_B_Mild/models/best_audio_model.pth",
        "Cand_C": "checkpoints/v16_1_Cand_C_Moderate/models/best_audio_model.pth",
        "Cand_D": "checkpoints/v16_1_Cand_D_Mixed/models/best_audio_model.pth"
    }
    
    noise_profiles = {
        "Clean": 0.0,
        "Mild": 0.5,
        "Moderate": 1.0,
        "Strong": 2.0
    }
    
    for c_name, ckpt in candidates.items():
        if not os.path.exists(ckpt):
            print(f"ERROR: Checkpoint missing for {c_name} at {ckpt}")
            continue
            
        print("="*60)
        print(f"Evaluating {c_name}")
        print("="*60)
        
        for n_name, n_level in noise_profiles.items():
            run_evaluation(c_name, ckpt, n_name, n_level)
            
    print("\nPhase 3 Validation Complete!")

if __name__ == "__main__":
    main()
