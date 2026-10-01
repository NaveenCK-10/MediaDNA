"""
V22.4 Phase 4: AVFF Baseline Re-evaluation
Re-runs the original VideoCAVMAEFT (AVFF) model with V14 checkpoint
on the DEV set to confirm its discrimination power.
Computes full metrics suite for head-to-head comparison.
"""
import os, sys, csv, json, time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, 
    confusion_matrix, average_precision_score)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT
import torchvision.transforms as T
import torchaudio
import soundfile as sf
import subprocess
from decord import VideoReader

FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")


def process_video(video_path):
    """Extract video frames and audio mel-spectrogram."""
    try:
        vr = VideoReader(video_path, width=224, height=224)
        num_frames = len(vr)
        frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
        v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        v_tensor = transform(v_tensor).permute(1, 0, 2, 3).unsqueeze(0)
        
        temp_wav = f"temp_avff_eval_{os.getpid()}.wav"
        cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        
        if not os.path.exists(temp_wav):
            a_tensor = torch.zeros(1, 1024, 128)
        else:
            wav, sr = sf.read(temp_wav)
            if len(wav.shape) > 1: wav = wav.mean(axis=1)
            audio_tensor = torch.FloatTensor(wav)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128
            )(audio_tensor)
            
            mel_spec = (mel_spec + 1e-6).log()
            mel_spec = (mel_spec - (-5.081)) / 4.4849
            
            target_len = 1024
            if mel_spec.shape[1] < target_len:
                pad = target_len - mel_spec.shape[1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
            else:
                mel_spec = mel_spec[:, :target_len]
            
            # torchaudio outputs (128, 1024). Model expects (T, D) = (1024, 128)
            mel_spec = mel_spec.transpose(0, 1)
            a_tensor = mel_spec.unsqueeze(0)
            os.remove(temp_wav)
        
        return a_tensor, v_tensor
    except Exception as e:
        print(f"  WARNING: Failed to process {video_path}: {e}")
        return None, None


def compute_metrics(labels, preds, preds_bin, conditions):
    """Compute full metrics suite."""
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0
    pr_auc = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0
    acc = accuracy_score(labels, preds_bin)
    bacc = balanced_accuracy_score(labels, preds_bin)
    prec = precision_score(labels, preds_bin, zero_division=0)
    rec = recall_score(labels, preds_bin, zero_division=0)
    f1 = f1_score(labels, preds_bin, zero_division=0)
    mcc = matthews_corrcoef(labels, preds_bin)
    
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds_bin).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    except:
        spec = 0
        tn = fp = fn = tp = 0
    
    return {
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "F1": f1,
        "MCC": mcc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def main():
    print("=" * 60)
    print("V22.4 PHASE 4: AVFF BASELINE RE-EVALUATION")
    print("=" * 60)
    
    # Load model
    print(f"Loading VideoCAVMAEFT from {CHECKPOINT}...")
    model = VideoCAVMAEFT()
    model = nn.DataParallel(model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    load_result = model.load_state_dict(ckpt, strict=False)
    print(f"  Missing keys: {len(load_result.missing_keys)}")
    print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")
    model.to(DEVICE)
    model.eval()
    
    # Load DEV manifest
    manifest = os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv")
    with open(manifest, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Evaluating on {len(rows)} DEV samples...")
    
    preds = []
    labels = []
    conditions = []
    paths = []
    errors = 0
    
    start = time.time()
    for i, row in enumerate(rows):
        fname = row["path"]
        dir_col = row.get("", "") or [v for k,v in row.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        
        rel_dir = dir_col.replace("FakeAVCeleb/", "")
        full_path = os.path.join(DATASET_DIR, rel_dir, fname)
        cat = row["type"]
        label = 0 if cat == "RealVideo-RealAudio" else 1
        
        cond = "RVRA"
        if cat == "RealVideo-FakeAudio": cond = "RVFA"
        elif cat == "FakeVideo-RealAudio": cond = "FVRA"
        elif cat == "FakeVideo-FakeAudio": cond = "FVFA"
        
        a_t, v_t = process_video(full_path)
        if a_t is None:
            errors += 1
            continue
        
        with torch.no_grad():
            a_t, v_t = a_t.to(DEVICE), v_t.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a_t, v_t)
            prob = float(torch.sigmoid(out).cpu().float().numpy()[0][0])
        
        preds.append(prob)
        labels.append(label)
        conditions.append(cond)
        paths.append(full_path)
        
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(rows) - i - 1)
            print(f"  [{i+1}/{len(rows)}] Elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s  Errors: {errors}")
    
    elapsed = time.time() - start
    print(f"\nEvaluation complete. {len(preds)} samples, {errors} errors, {elapsed:.0f}s")
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    # Threshold sweep
    best_mcc = -1
    best_thresh = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        p_bin = (preds >= t).astype(int)
        mcc = matthews_corrcoef(labels, p_bin)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
    
    print(f"\nOptimal MCC threshold: {best_thresh:.2f} (MCC={best_mcc:.4f})")
    
    # Compute metrics at optimal threshold
    preds_bin = (preds >= best_thresh).astype(int)
    metrics_optimal = compute_metrics(labels, preds, preds_bin, conditions)
    
    # Compute metrics at threshold 0.5
    preds_bin_05 = (preds >= 0.5).astype(int)
    metrics_05 = compute_metrics(labels, preds, preds_bin_05, conditions)
    
    # Score distribution by condition
    score_dist = {}
    for cond in ["RVRA", "RVFA", "FVRA", "FVFA"]:
        c_scores = [p for i, p in enumerate(preds) if conditions[i] == cond]
        if c_scores:
            score_dist[cond] = {
                "N": len(c_scores),
                "mean": float(np.mean(c_scores)),
                "median": float(np.median(c_scores)),
                "std": float(np.std(c_scores)),
                "min": float(np.min(c_scores)),
                "max": float(np.max(c_scores))
            }
    
    # Real vs Fake
    real_scores = preds[labels == 0]
    fake_scores = preds[labels == 1]
    
    print("\n" + "=" * 60)
    print("AVFF BASELINE DEV RESULTS")
    print("=" * 60)
    print(f"ROC-AUC:           {metrics_optimal['ROC_AUC']:.4f}")
    print(f"PR-AUC:            {metrics_optimal['PR_AUC']:.4f}")
    print(f"Balanced Accuracy: {metrics_optimal['balanced_accuracy']:.4f}")
    print(f"MCC:               {metrics_optimal['MCC']:.4f}")
    print(f"Precision:         {metrics_optimal['precision']:.4f}")
    print(f"Recall:            {metrics_optimal['recall']:.4f}")
    print(f"Specificity:       {metrics_optimal['specificity']:.4f}")
    print(f"F1:                {metrics_optimal['F1']:.4f}")
    print(f"\nReal scores:  mean={real_scores.mean():.4f}  std={real_scores.std():.4f}  range=[{real_scores.min():.4f}, {real_scores.max():.4f}]")
    print(f"Fake scores:  mean={fake_scores.mean():.4f}  std={fake_scores.std():.4f}  range=[{fake_scores.min():.4f}, {fake_scores.max():.4f}]")
    print(f"\nCondition breakdown:")
    for cond, s in score_dist.items():
        print(f"  {cond}: N={s['N']:4d}  mean={s['mean']:.4f}  median={s['median']:.4f}")
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "VideoCAVMAEFT (AVFF)",
        "checkpoint": "v14_fullscale/best_audio_model.pth",
        "dataset": "v22_2_dev.csv",
        "n_samples": len(preds),
        "n_errors": errors,
        "eval_time_seconds": elapsed,
        "optimal_threshold": float(best_thresh),
        "metrics_at_optimal_threshold": metrics_optimal,
        "metrics_at_threshold_0.5": metrics_05,
        "score_distribution": score_dist,
        "real_score_stats": {
            "mean": float(real_scores.mean()),
            "std": float(real_scores.std()),
            "min": float(real_scores.min()),
            "max": float(real_scores.max())
        },
        "fake_score_stats": {
            "mean": float(fake_scores.mean()),
            "std": float(fake_scores.std()),
            "min": float(fake_scores.min()),
            "max": float(fake_scores.max())
        }
    }
    
    out_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "avff_baseline_dev_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Save predictions CSV
    pred_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "avff_baseline_dev_predictions.csv")
    with open(pred_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "condition", "probability", "prediction"])
        for p_val, l_val, c_val, path in zip(preds, labels, conditions, paths):
            p_bin = 1 if p_val >= best_thresh else 0
            writer.writerow([os.path.basename(path), int(l_val), c_val, f"{p_val:.6f}", p_bin])
    
    print(f"\nResults saved to {out_path}")
    print(f"Predictions saved to {pred_path}")

if __name__ == "__main__":
    main()
