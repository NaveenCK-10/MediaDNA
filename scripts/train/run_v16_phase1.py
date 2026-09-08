import os
import subprocess

def run_evaluation(exp_name, noise_level):
    print(f"Running {exp_name} with noise level {noise_level}...")
    cmd = [
        "python", "v16_experiment_harness.py",
        "--exp_name", exp_name,
        "--mode", "normal",
        "--noise_level", str(noise_level),
        "--csv", "data/val_v14.csv"
    ]
    subprocess.run(cmd, check=True)

def main():
    run_evaluation("v16_phase1_A_clean", 0.0)
    run_evaluation("v16_phase1_B_mild", 0.5)
    run_evaluation("v16_phase1_C_moderate", 1.0)
    run_evaluation("v16_phase1_D_strong", 2.0)
    
    print("Phase 1 complete!")

if __name__ == "__main__":
    main()
