import os
import csv
import random
import subprocess
import torch
import hashlib
import sys

FAKEAVCELEB_ROOT = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
TRAIN_CSV = "data/train_smoke.csv"
VAL_CSV = "data/val_smoke.csv"
INITIAL_CHECKPOINT = r"checkpoints\stage-3.pth"
SAVE_DIR = r"exp\stage-3-local"
BEST_CHECKPOINT = os.path.join(SAVE_DIR, r"models\best_audio_model.pth")

def create_subset_csv(filepath, num_fake, num_real, used_videos=None):
    if used_videos is None:
        used_videos = set()
    
    fake_dirs = ['FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio']
    real_dirs = ['RealVideo-RealAudio']
    
    fake_videos = []
    real_videos = []
    
    for cdir in fake_dirs:
        full_dir = os.path.join(FAKEAVCELEB_ROOT, cdir)
        if os.path.exists(full_dir):
            for root, _, files in os.walk(full_dir):
                for f in files:
                    if f.endswith('.mp4'):
                        vpath = os.path.join(root, f)
                        if vpath not in used_videos:
                            fake_videos.append(vpath)
                            
    for cdir in real_dirs:
        full_dir = os.path.join(FAKEAVCELEB_ROOT, cdir)
        if os.path.exists(full_dir):
            for root, _, files in os.walk(full_dir):
                for f in files:
                    if f.endswith('.mp4'):
                        vpath = os.path.join(root, f)
                        if vpath not in used_videos:
                            real_videos.append(vpath)

    random.seed(42)
    selected_fake = random.sample(fake_videos, min(num_fake, len(fake_videos)))
    selected_real = random.sample(real_videos, min(num_real, len(real_videos)))
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label'])
        for v in selected_fake:
            writer.writerow([v, 1])
            used_videos.add(v)
        for v in selected_real:
            writer.writerow([v, 0])
            used_videos.add(v)
            
    print(f"Created {filepath} with {len(selected_fake)} Fake and {len(selected_real)} Real samples.")
    return used_videos

def get_weight_stats(ckpt_path):
    print(f"Reading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    stats = {}
    for key in ['module.mlp_head.fc1.weight', 'module.mlp_head.fc2.weight', 'module.mlp_head.fc3.weight']:
        if key in ckpt:
            tensor = ckpt[key].float()
            std = tensor.std().item()
            mean = tensor.mean().item()
            # simple hash approximation
            thash = hashlib.md5(tensor.numpy().tobytes()).hexdigest()[:8]
            stats[key] = {'std': std, 'mean': mean, 'hash': thash}
        else:
            stats[key] = None
    return stats

def main():
    print("="*60)
    print("1. Preparing Datasets")
    print("="*60)
    os.makedirs('data', exist_ok=True)
    used = create_subset_csv(TRAIN_CSV, 20, 20)
    create_subset_csv(VAL_CSV, 10, 10, used)
    
    print("\n" + "="*60)
    print("2. Recording Initial Classifier Weights")
    print("="*60)
    initial_stats = get_weight_stats(INITIAL_CHECKPOINT)
    for k, v in initial_stats.items():
        if v:
            print(f"{k} -> std: {v['std']:.6f}, mean: {v['mean']:.6f}, hash: {v['hash']}")
            
    print("\n" + "="*60)
    print("3. Executing Local Stage-3 Fine-Tuning")
    print("="*60)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'models'), exist_ok=True)
    
    cmd = [
        sys.executable, "-W", "ignore", r"src\run_ft.py",
        "--data-train", TRAIN_CSV,
        "--data-val", VAL_CSV,
        "--save-dir", SAVE_DIR,
        "--n_classes", "2",
        "--lr", "1e-5",
        "--head_lr", "50",
        "--n-epochs", "2",
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
    
    print("Running command:")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return
        
    print("\n" + "="*60)
    print("4. Verifying Classifier Weight Updates")
    print("="*60)
    
    if not os.path.exists(BEST_CHECKPOINT):
        print(f"ERROR: {BEST_CHECKPOINT} was not created!")
        return
        
    final_stats = get_weight_stats(BEST_CHECKPOINT)
    
    changed = False
    for k in initial_stats.keys():
        if initial_stats[k] and final_stats[k]:
            i_hash = initial_stats[k]['hash']
            f_hash = final_stats[k]['hash']
            i_std = initial_stats[k]['std']
            f_std = final_stats[k]['std']
            if i_hash != f_hash:
                print(f"SUCCESS: {k} changed! (Hash: {i_hash} -> {f_hash}, Std: {i_std:.6f} -> {f_std:.6f})")
                changed = True
            else:
                print(f"WARNING: {k} did NOT change! (Hash: {i_hash})")
                
    if changed:
        print("\nCONCLUSION: The classifier weights successfully updated during training.")
    else:
        print("\nCONCLUSION: The classifier weights did NOT update. Check learning rate or gradient flow.")

if __name__ == '__main__':
    main()
