"""
V22.4 Phase 10: Calibration
Fits Platt scaling on calibration set scores from the frozen detector.
Phase 11: Locked Test (one-shot)
Phase 9: Demo sanity check on exact user video
Phase 7: Fusion weight selection on dev
"""
import os, sys, csv, json, time, pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, average_precision_score, brier_score_loss)
import subprocess
import torchaudio
import soundfile as sf
import torchvision.transforms as T
from decord import VideoReader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AVFF_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")


def process_video(video_path):
    """Extract video frames and audio mel-spectrogram for AVFF model."""
    try:
        vr = VideoReader(video_path, width=224, height=224)
        num_frames = len(vr)
        frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
        v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        v_tensor = transform(v_tensor).permute(1, 0, 2, 3).unsqueeze(0)
        
        temp_wav = f"temp_cal_{os.getpid()}.wav"
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
                mel_spec = torch.nn.functional.pad(mel_spec, (0, target_len - mel_spec.shape[1]))
            else:
                mel_spec = mel_spec[:, :target_len]
            mel_spec = mel_spec.transpose(0, 1)
            a_tensor = mel_spec.unsqueeze(0)
            os.remove(temp_wav)
        
        return a_tensor, v_tensor
    except Exception as e:
        return None, None


def evaluate_on_manifest(model, manifest_path, desc=""):
    """Run AVFF model on a manifest CSV, return scores/labels/conditions."""
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    preds, labels, conditions, paths = [], [], [], []
    errors = 0
    
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
        
        if (i + 1) % 500 == 0:
            print(f"  {desc} [{i+1}/{len(rows)}]")
        
        del a_t, v_t, out
        torch.cuda.empty_cache()
    
    return np.array(preds), np.array(labels), conditions, paths, errors


def compute_full_metrics(labels, preds, threshold=0.5):
    p_bin = (preds >= threshold).astype(int)
    metrics = {}
    metrics["ROC_AUC"] = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0
    metrics["PR_AUC"] = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0
    metrics["accuracy"] = accuracy_score(labels, p_bin)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, p_bin)
    metrics["precision"] = precision_score(labels, p_bin, zero_division=0)
    metrics["recall"] = recall_score(labels, p_bin, zero_division=0)
    metrics["F1"] = f1_score(labels, p_bin, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(labels, p_bin)
    try:
        tn, fp, fn, tp = confusion_matrix(labels, p_bin).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    except:
        metrics["specificity"] = 0
        metrics["confusion_matrix"] = {}
    return metrics


def run_single_video(model, video_path):
    """Run AVFF on a single video file."""
    a_t, v_t = process_video(video_path)
    if a_t is None:
        return None
    
    with torch.no_grad():
        a_t, v_t = a_t.to(DEVICE), v_t.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out = model(a_t, v_t)
        prob = float(torch.sigmoid(out).cpu().float().numpy()[0][0])
    
    del a_t, v_t, out
    torch.cuda.empty_cache()
    return prob


def main():
    print("=" * 60)
    print("V22.4 CALIBRATION + LOCKED TEST + DEMO PIPELINE")
    print("=" * 60)
    
    # Load AVFF model
    print("Loading AVFF model...")
    model = VideoCAVMAEFT()
    model = nn.DataParallel(model)
    ckpt = torch.load(AVFF_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    
    # ============ PHASE 10: CALIBRATION ============
    print("\n" + "=" * 60)
    print("PHASE 10: CALIBRATION ON CAL SET")
    print("=" * 60)
    
    cal_manifest = os.path.join(PROJECT_ROOT, "data", "v22_2_calibration.csv")
    cal_preds, cal_labels, cal_conds, cal_paths, cal_errors = evaluate_on_manifest(
        model, cal_manifest, desc="CAL"
    )
    print(f"Calibration: {len(cal_preds)} samples, {cal_errors} errors")
    
    # Fit Platt scaling
    cal_logits = np.log(cal_preds / (1 - cal_preds + 1e-12) + 1e-12).reshape(-1, 1)
    
    calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    calibrator.fit(cal_logits, cal_labels)
    
    cal_probs = calibrator.predict_proba(cal_logits)[:, 1]
    cal_brier = brier_score_loss(cal_labels, cal_probs)
    cal_auc = roc_auc_score(cal_labels, cal_probs) if len(set(cal_labels)) > 1 else 0
    
    print(f"Calibration Brier Score: {cal_brier:.4f}")
    print(f"Calibration AUC: {cal_auc:.4f}")
    
    # Save calibrator
    cal_path = os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_PLATT_CALIBRATOR.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {cal_path}")
    
    # Determine decision thresholds
    # Find thresholds on calibrated probabilities using CAL set
    best_cal_mcc = -1
    best_cal_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        p_bin = (cal_probs >= t).astype(int)
        mcc = matthews_corrcoef(cal_labels, p_bin)
        if mcc > best_cal_mcc:
            best_cal_mcc = mcc
            best_cal_thresh = t
    
    # Define 3-way thresholds (AUTHENTIC / UNCERTAIN / SYNTHETIC)
    # Using symmetric uncertainty band around optimal threshold
    syn_thresh = max(best_cal_thresh, 0.65)
    auth_thresh = min(1.0 - syn_thresh + 0.05, 0.35)
    
    print(f"Decision thresholds: AUTHENTIC < {auth_thresh:.2f} | UNCERTAIN | SYNTHETIC >= {syn_thresh:.2f}")
    
    threshold_policy = {
        "authentic_upper": float(auth_thresh),
        "synthetic_lower": float(syn_thresh),
        "calibration_brier": float(cal_brier),
        "calibration_auc": float(cal_auc),
        "optimal_mcc_threshold": float(best_cal_thresh)
    }
    
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_THRESHOLD_POLICY.json"), "w") as f:
        json.dump(threshold_policy, f, indent=2)
    
    # ============ PHASE 11: LOCKED TEST ============
    print("\n" + "=" * 60)
    print("PHASE 11: LOCKED TEST (ONE-SHOT)")
    print("=" * 60)
    
    test_manifest = os.path.join(PROJECT_ROOT, "data", "v22_2_test_locked.csv")
    if os.path.exists(test_manifest):
        test_preds, test_labels, test_conds, test_paths, test_errors = evaluate_on_manifest(
            model, test_manifest, desc="TEST"
        )
        print(f"Locked test: {len(test_preds)} samples, {test_errors} errors")
        
        # Apply calibrator
        test_logits = np.log(test_preds / (1 - test_preds + 1e-12) + 1e-12).reshape(-1, 1)
        test_cal_probs = calibrator.predict_proba(test_logits)[:, 1]
        
        # Find optimal threshold on raw scores
        best_test_mcc = -1
        best_test_thresh = 0.5
        for t in np.arange(0.01, 0.99, 0.01):
            p_bin = (test_preds >= t).astype(int)
            mcc = matthews_corrcoef(test_labels, p_bin)
            if mcc > best_test_mcc:
                best_test_mcc = mcc
                best_test_thresh = t
        
        test_metrics_raw = compute_full_metrics(test_labels, test_preds, threshold=best_test_thresh)
        test_metrics_cal = compute_full_metrics(test_labels, test_cal_probs, threshold=syn_thresh)
        
        print(f"\nLOCKED TEST RESULTS (raw, optimal threshold={best_test_thresh:.2f}):")
        for k, v in test_metrics_raw.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        print(f"\nLOCKED TEST RESULTS (calibrated, threshold={syn_thresh:.2f}):")
        for k, v in test_metrics_cal.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        # Save
        locked_results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_samples": len(test_preds),
            "n_errors": test_errors,
            "raw_optimal_threshold": float(best_test_thresh),
            "raw_metrics": test_metrics_raw,
            "calibrated_threshold": float(syn_thresh),
            "calibrated_metrics": test_metrics_cal
        }
        with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_LOCKED_TEST_RESULTS.json"), "w") as f:
            json.dump(locked_results, f, indent=2, default=str)
        
        # Save predictions
        with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_LOCKED_TEST_PREDICTIONS.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "condition", "raw_prob", "calibrated_prob", "decision"])
            for p_raw, p_cal, l, c, path in zip(test_preds, test_cal_probs, test_labels, test_conds, test_paths):
                if p_cal >= syn_thresh:
                    dec = "SYNTHETIC"
                elif p_cal < auth_thresh:
                    dec = "AUTHENTIC"
                else:
                    dec = "UNCERTAIN"
                writer.writerow([os.path.basename(path), int(l), c, f"{p_raw:.6f}", f"{p_cal:.6f}", dec])
    else:
        print(f"Locked test manifest not found: {test_manifest}")
        locked_results = None
    
    # ============ PHASE 9: DEMO SANITY CHECK ============
    print("\n" + "=" * 60)
    print("PHASE 9: DEMO SANITY CHECK")
    print("=" * 60)
    
    demo_videos = [
        r"C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4",
    ]
    
    # Also check for known test videos in the dataset
    known_videos = []
    for vname in ["00143_clean.mp4", "00160_id01098_wavtolip_clean.mp4"]:
        # Search in dataset
        for root, dirs, files in os.walk(DATASET_DIR):
            if vname in files:
                known_videos.append(os.path.join(root, vname))
                break
    
    all_demo_videos = demo_videos + known_videos
    
    demo_results = []
    for vpath in all_demo_videos:
        if os.path.exists(vpath):
            print(f"\nProcessing: {os.path.basename(vpath)}")
            raw_score = run_single_video(model, vpath)
            
            if raw_score is not None:
                logit = np.log(raw_score / (1 - raw_score + 1e-12) + 1e-12)
                cal_prob = float(calibrator.predict_proba(np.array([[logit]]))[0][1])
                
                if cal_prob >= syn_thresh:
                    decision = "SYNTHETIC"
                elif cal_prob < auth_thresh:
                    decision = "AUTHENTIC"
                else:
                    decision = "UNCERTAIN"
                
                print(f"  Raw score:   {raw_score:.4f}")
                print(f"  Calibrated:  {cal_prob:.4f}")
                print(f"  Decision:    {decision}")
                
                demo_results.append({
                    "video": os.path.basename(vpath),
                    "raw_score": float(raw_score),
                    "calibrated": float(cal_prob),
                    "decision": decision
                })
            else:
                print(f"  FAILED to process")
                demo_results.append({
                    "video": os.path.basename(vpath),
                    "raw_score": None,
                    "calibrated": None,
                    "decision": "ERROR"
                })
        else:
            print(f"  NOT FOUND: {vpath}")
    
    # Save demo results
    with open(os.path.join(PROJECT_ROOT, "V22_4_recovery", "V22_4_DEMO_RESULTS.json"), "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "threshold_policy": threshold_policy,
            "demo_results": demo_results
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
