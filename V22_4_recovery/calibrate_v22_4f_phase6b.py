"""
PHASE 6B: Balanced Calibration Experiment
==========================================
1. Extract raw scores from V22.4F StageB_Ep2 on DEV and CAL (one-time, saved to disk)
2. Fit unweighted Platt calibrator (audit copy)
3. Fit class-balanced Platt calibrator
4. Evaluate candidate 3-state policies on DEV for both
5. Compute calibration diagnostics (Brier, ECE, reliability curve data)
6. Save all artifacts

NOTE: The raw scores were not persisted from the Phase 6 run.
      This script performs ONE final inference pass to extract and save them.
      After this, no further neural-network inference is needed for calibration work.
"""

import os, sys, csv, time, json, pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
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
SAVE_DIR = os.path.join(PROJECT_ROOT, "V22_4_recovery")

# ── Dataset ──────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            self.rows = list(csv.DictReader(f))
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.labels = []
        for r in self.rows:
            self.labels.append(0 if r["type"] == "RealVideo-RealAudio" else 1)

    def __len__(self): return len(self.rows)

    def _extract_mel(self, video_path):
        temp_wav = f"temp_aud_{os.getpid()}_{id(self)}_{time.time()}.wav"
        try:
            cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            if not os.path.exists(temp_wav): return torch.zeros(1024, 128)
            wav, sr = sf.read(temp_wav)
            if len(wav.shape) > 1: wav = wav.mean(axis=1)
            audio_tensor = torch.FloatTensor(wav)
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128)(audio_tensor)
            mel_spec = (mel_spec + 1e-6).log()
            mel_spec = (mel_spec - (-5.081)) / 4.4849
            target_len = 1024
            if mel_spec.shape[1] < target_len:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, target_len - mel_spec.shape[1]))
            else:
                mel_spec = mel_spec[:, :target_len]
            mel_spec = mel_spec.transpose(0, 1)
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return mel_spec
        except:
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return torch.zeros(1024, 128)

    def __getitem__(self, idx):
        r = self.rows[idx]
        fname = r["path"]
        dir_col = r.get("", "") or [v for k,v in r.items() if k is None or k == ""][0]
        if isinstance(dir_col, list): dir_col = dir_col[0]
        full_path = os.path.join(DATASET_DIR, dir_col.replace("FakeAVCeleb/", ""), fname)
        label = 0 if r["type"] == "RealVideo-RealAudio" else 1
        try:
            vr = VideoReader(full_path, width=224, height=224)
            frame_idx = np.linspace(0, len(vr) - 1, 16, dtype=int)
            frames = vr.get_batch(frame_idx).asnumpy()
            v_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            v_t = self.transform(v_t).permute(1, 0, 2, 3)
        except:
            v_t = torch.zeros(3, 16, 224, 224)
        a_t = self._extract_mel(full_path)
        return a_t, v_t, torch.tensor([label], dtype=torch.float32), full_path

# ── Inference ────────────────────────────────────────────────────────────────

def extract_raw_scores(model, dataloader, name):
    model.eval()
    preds, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (a, v, l, _) in enumerate(dataloader):
            if i % 200 == 0: print(f"  {name} progress: {i}/{len(dataloader)} batches")
            a, v = a.to(DEVICE), v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            preds.extend(prob)
            labels.extend(l.numpy()[:, 0])
    dur = time.time() - t0
    preds = np.array(preds, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    print(f"  {name} done: {len(preds)} samples, {dur:.1f}s")
    return preds, labels

# ── Calibration diagnostics ──────────────────────────────────────────────────

def compute_calibration_diagnostics(labels, probs, name):
    """Compute Brier score, ECE, and reliability-curve bin data."""
    brier = brier_score_loss(labels, probs)

    # ECE (Expected Calibration Error) with 10 bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_data.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": 0,
                             "mean_pred": None, "mean_true": None, "gap": None})
            continue
        mean_pred = probs[mask].mean()
        mean_true = labels[mask].mean()
        gap = abs(mean_pred - mean_true)
        ece += (n_in_bin / len(probs)) * gap
        bin_data.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(n_in_bin),
                         "mean_pred": round(float(mean_pred), 4),
                         "mean_true": round(float(mean_true), 4),
                         "gap": round(float(gap), 4)})

    print(f"\n--- {name} Calibration Diagnostics ---")
    print(f"  Brier Score: {brier:.6f}")
    print(f"  ECE (10 bins): {ece:.6f}")
    for b in bin_data:
        if b["n"] > 0:
            print(f"    {b['bin']}  n={b['n']:>5}  pred={b['mean_pred']:.4f}  true={b['mean_true']:.4f}  gap={b['gap']:.4f}")

    return {"brier": round(float(brier), 6), "ece": round(float(ece), 6), "bins": bin_data}

# ── 3-state policy search ────────────────────────────────────────────────────

def search_policies(scores, labels, name):
    """
    Search over candidate 3-state thresholds on DEV.
    States:
        AUTHENTIC  if score < t_low
        SUSPECT    if t_low <= score <= t_high
        SYNTHETIC  if score > t_high
    """
    print(f"\n{'='*60}")
    print(f"3-STATE POLICY SEARCH: {name}")
    print(f"{'='*60}")

    candidates = []
    t_lows  = np.arange(0.10, 0.55, 0.01)
    t_highs = np.arange(0.45, 0.96, 0.01)

    for t_lo in t_lows:
        for t_hi in t_highs:
            if t_lo >= t_hi:
                continue

            # Assign states
            pred_auth = scores < t_lo
            pred_fake = scores > t_hi
            pred_suspect = ~pred_auth & ~pred_fake

            n_auth = pred_auth.sum()
            n_fake = pred_fake.sum()
            n_suspect = pred_suspect.sum()
            abstention_rate = n_suspect / len(scores)

            # For binary metrics, evaluate only the confident predictions
            confident_mask = pred_auth | pred_fake
            if confident_mask.sum() < 10:
                continue

            y_conf = labels[confident_mask].astype(int)
            p_conf = pred_fake[confident_mask].astype(int)  # 1 = predicted fake

            if len(np.unique(y_conf)) < 2 or len(np.unique(p_conf)) < 2:
                continue

            cm = confusion_matrix(y_conf, p_conf)
            if cm.shape != (2, 2):
                continue

            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            mcc  = matthews_corrcoef(y_conf, p_conf)
            bacc = balanced_accuracy_score(y_conf, p_conf)

            candidates.append({
                "t_low": round(float(t_lo), 2),
                "t_high": round(float(t_hi), 2),
                "abstention_rate": round(float(abstention_rate), 4),
                "mcc": round(float(mcc), 4),
                "bacc": round(float(bacc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "specificity": round(float(spec), 4),
                "f1": round(float(f1), 4),
                "cm": cm.tolist(),
                "n_auth": int(n_auth),
                "n_suspect": int(n_suspect),
                "n_fake": int(n_fake),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })

    if not candidates:
        print("  WARNING: No valid 3-state candidates found!")
        return None, candidates

    # Sort by MCC descending, then by abstention_rate ascending (prefer lower abstention)
    candidates.sort(key=lambda c: (-c["mcc"], c["abstention_rate"]))

    # Print top 10
    print(f"\nTop 10 candidate policies (of {len(candidates)} searched):")
    print(f"{'t_low':>6} {'t_high':>6} {'Abst%':>6} {'MCC':>7} {'BAcc':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6} {'Pred_R':>6} {'Pred_F':>6}")
    print("-" * 80)
    for c in candidates[:10]:
        print(f"{c['t_low']:>6.2f} {c['t_high']:>6.2f} {c['abstention_rate']*100:>5.1f}% "
              f"{c['mcc']:>7.4f} {c['bacc']:>6.4f} {c['precision']:>6.4f} {c['recall']:>6.4f} "
              f"{c['specificity']:>6.4f} {c['f1']:>6.4f} {c['n_auth']:>6} {c['n_fake']:>6}")

    best = candidates[0]
    print(f"\nSELECTED POLICY: t_low={best['t_low']}, t_high={best['t_high']}")
    print(f"  MCC={best['mcc']}, BAcc={best['bacc']}, Abstention={best['abstention_rate']*100:.1f}%")
    print(f"  Precision={best['precision']}, Recall={best['recall']}, Specificity={best['specificity']}")
    print(f"  Confusion Matrix (confident only): {best['cm']}")
    print(f"  Predictions: {best['n_auth']} Authentic / {best['n_suspect']} Suspect / {best['n_fake']} Synthetic")

    return best, candidates

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 6B: BALANCED CALIBRATION EXPERIMENT")
    print("=" * 60)

    # ── Step 1: Load or extract raw scores ───────────────────────────────────

    dev_raw_path = os.path.join(SAVE_DIR, "V22_4F_DEV_raw_scores.npz")
    cal_raw_path = os.path.join(SAVE_DIR, "V22_4F_CAL_raw_scores.npz")

    if os.path.exists(dev_raw_path) and os.path.exists(cal_raw_path):
        print("\nLoading persisted raw scores (no inference needed)...")
        dev_data = np.load(dev_raw_path)
        dev_preds, dev_labels = dev_data["preds"], dev_data["labels"]
        cal_data = np.load(cal_raw_path)
        cal_preds, cal_labels = cal_data["preds"], cal_data["labels"]
        print(f"  DEV: {len(dev_preds)} scores loaded")
        print(f"  CAL: {len(cal_preds)} scores loaded")
    else:
        print("\nRaw scores not found on disk. Running ONE-TIME inference to extract and persist them...")
        ckpt_path = os.path.join(SAVE_DIR, "V22_4F_MULTIMODAL_StageB_Ep2.pth")
        model = VideoCAVMAEFT()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        model.to(DEVICE)
        print("  Loaded StageB_Ep2 checkpoint.")

        dev_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_dev.csv"))
        dev_loader = DataLoader(dev_dataset, batch_size=2, shuffle=False, num_workers=0)
        dev_preds, dev_labels = extract_raw_scores(model, dev_loader, "DEV")
        np.savez(dev_raw_path, preds=dev_preds, labels=dev_labels)
        print(f"  Saved DEV raw scores to {dev_raw_path}")

        cal_dataset = MultimodalDataset(os.path.join(PROJECT_ROOT, "data", "v22_2_calibration.csv"))
        cal_loader = DataLoader(cal_dataset, batch_size=2, shuffle=False, num_workers=0)
        cal_preds, cal_labels = extract_raw_scores(model, cal_loader, "CAL")
        np.savez(cal_raw_path, preds=cal_preds, labels=cal_labels)
        print(f"  Saved CAL raw scores to {cal_raw_path}")

        del model
        torch.cuda.empty_cache()

    # ── Verify DEV reproducibility ───────────────────────────────────────────
    print("\n--- DEV Reproducibility Check ---")
    p_bin = (dev_preds >= 0.5).astype(int)
    auc = roc_auc_score(dev_labels, dev_preds)
    mcc = matthews_corrcoef(dev_labels.astype(int), p_bin)
    print(f"  ROC-AUC: {auc:.4f} (expected 0.7292)")
    print(f"  MCC:     {mcc:.4f} (expected 0.1436)")
    assert abs(auc - 0.7292) < 0.005, f"AUC mismatch: {auc}"
    assert abs(mcc - 0.1436) < 0.005, f"MCC mismatch: {mcc}"
    print("  PASSED.\n")

    # ── Raw score statistics ─────────────────────────────────────────────────
    print("--- Raw Score Distribution ---")
    print(f"  DEV: mean={dev_preds.mean():.4f}, std={dev_preds.std():.4f}, "
          f"min={dev_preds.min():.4f}, max={dev_preds.max():.4f}")
    print(f"  CAL: mean={cal_preds.mean():.4f}, std={cal_preds.std():.4f}, "
          f"min={cal_preds.min():.4f}, max={cal_preds.max():.4f}")

    dev_real_mask = dev_labels == 0
    dev_fake_mask = dev_labels == 1
    print(f"  DEV Real  (n={dev_real_mask.sum()}): mean={dev_preds[dev_real_mask].mean():.4f}, std={dev_preds[dev_real_mask].std():.4f}")
    print(f"  DEV Fake  (n={dev_fake_mask.sum()}): mean={dev_preds[dev_fake_mask].mean():.4f}, std={dev_preds[dev_fake_mask].std():.4f}")

    cal_real_mask = cal_labels == 0
    cal_fake_mask = cal_labels == 1
    print(f"  CAL Real  (n={cal_real_mask.sum()}): mean={cal_preds[cal_real_mask].mean():.4f}, std={cal_preds[cal_real_mask].std():.4f}")
    print(f"  CAL Fake  (n={cal_fake_mask.sum()}): mean={cal_preds[cal_fake_mask].mean():.4f}, std={cal_preds[cal_fake_mask].std():.4f}")

    # ── A) Unweighted Platt Calibrator ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("CALIBRATOR A: Unweighted Platt Scaling")
    print("=" * 60)

    cal_A = LogisticRegression(solver="lbfgs")
    cal_A.fit(cal_preds.reshape(-1, 1), cal_labels.astype(int))

    dev_cal_A = cal_A.predict_proba(dev_preds.reshape(-1, 1))[:, 1]
    print(f"  Coef: {cal_A.coef_[0][0]:.6f}, Intercept: {cal_A.intercept_[0]:.6f}")
    print(f"  DEV calibrated scores: mean={dev_cal_A.mean():.4f}, std={dev_cal_A.std():.4f}, "
          f"min={dev_cal_A.min():.4f}, max={dev_cal_A.max():.4f}")

    diag_A = compute_calibration_diagnostics(dev_labels, dev_cal_A, "Calibrator A (Unweighted)")
    best_A, candidates_A = search_policies(dev_cal_A, dev_labels, "Calibrator A (Unweighted)")

    cal_A_path = os.path.join(SAVE_DIR, "V22_4F_PLATT_CALIBRATOR_A_unweighted.pkl")
    with open(cal_A_path, "wb") as f:
        pickle.dump(cal_A, f)
    print(f"  Saved: {cal_A_path}")

    # ── B) Class-Balanced Platt Calibrator ───────────────────────────────────
    print("\n" + "=" * 60)
    print("CALIBRATOR B: Class-Balanced Platt Scaling")
    print("=" * 60)

    cal_B = LogisticRegression(class_weight="balanced", solver="lbfgs")
    cal_B.fit(cal_preds.reshape(-1, 1), cal_labels.astype(int))

    dev_cal_B = cal_B.predict_proba(dev_preds.reshape(-1, 1))[:, 1]
    print(f"  Coef: {cal_B.coef_[0][0]:.6f}, Intercept: {cal_B.intercept_[0]:.6f}")
    print(f"  DEV calibrated scores: mean={dev_cal_B.mean():.4f}, std={dev_cal_B.std():.4f}, "
          f"min={dev_cal_B.min():.4f}, max={dev_cal_B.max():.4f}")

    diag_B = compute_calibration_diagnostics(dev_labels, dev_cal_B, "Calibrator B (Balanced)")
    best_B, candidates_B = search_policies(dev_cal_B, dev_labels, "Calibrator B (Balanced)")

    cal_B_path = os.path.join(SAVE_DIR, "V22_4F_PLATT_CALIBRATOR_B_balanced.pkl")
    with open(cal_B_path, "wb") as f:
        pickle.dump(cal_B, f)
    print(f"  Saved: {cal_B_path}")

    # ── C) Raw scores (no calibrator) policy search ──────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE: Raw Sigmoid Scores (No Calibrator)")
    print("=" * 60)

    diag_raw = compute_calibration_diagnostics(dev_labels, dev_preds, "Raw Sigmoid Scores")
    best_raw, candidates_raw = search_policies(dev_preds, dev_labels, "Raw Sigmoid Scores")

    # ── Save full comparison report ──────────────────────────────────────────
    report = {
        "calibrator_A_unweighted": {
            "coef": float(cal_A.coef_[0][0]),
            "intercept": float(cal_A.intercept_[0]),
            "diagnostics": diag_A,
            "best_policy": best_A,
            "n_candidates_searched": len(candidates_A),
        },
        "calibrator_B_balanced": {
            "coef": float(cal_B.coef_[0][0]),
            "intercept": float(cal_B.intercept_[0]),
            "diagnostics": diag_B,
            "best_policy": best_B,
            "n_candidates_searched": len(candidates_B),
        },
        "raw_sigmoid": {
            "diagnostics": diag_raw,
            "best_policy": best_raw,
            "n_candidates_searched": len(candidates_raw),
        },
    }

    report_path = os.path.join(SAVE_DIR, "V22_4F_CALIBRATION_COMPARISON.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nSaved full comparison report to {report_path}")

    # ── Final comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    rows = []
    for label, best, diag in [
        ("Raw Sigmoid", best_raw, diag_raw),
        ("Platt Unweighted", best_A, diag_A),
        ("Platt Balanced", best_B, diag_B),
    ]:
        if best:
            rows.append((label, best, diag))

    print(f"\n{'Method':<20} {'t_low':>6} {'t_high':>6} {'MCC':>7} {'BAcc':>6} "
          f"{'Prec':>6} {'Rec':>6} {'Spec':>6} {'Abst%':>6} {'Brier':>8} {'ECE':>8}")
    print("-" * 100)
    for label, best, diag in rows:
        print(f"{label:<20} {best['t_low']:>6.2f} {best['t_high']:>6.2f} "
              f"{best['mcc']:>7.4f} {best['bacc']:>6.4f} {best['precision']:>6.4f} "
              f"{best['recall']:>6.4f} {best['specificity']:>6.4f} "
              f"{best['abstention_rate']*100:>5.1f}% {diag['brier']:>8.6f} {diag['ece']:>8.6f}")

    # ── Select winner ────────────────────────────────────────────────────────
    valid = [(l, b, d) for l, b, d in rows if b["mcc"] > 0]
    if valid:
        winner = max(valid, key=lambda x: x[1]["mcc"])
        print(f"\nWINNER: {winner[0]} (MCC={winner[1]['mcc']:.4f})")
    else:
        print("\nWARNING: No calibration method produced MCC > 0.")

    # ── Save the winning policy ──────────────────────────────────────────────
    if valid:
        w_label, w_best, w_diag = winner
        policy = {
            "model": "V22_4F_Multimodal",
            "checkpoint": "V22_4F_MULTIMODAL_StageB_Ep2.pth",
            "calibration_method": w_label,
            "output_label": "class-balanced decision score" if "Balanced" in w_label else "sigmoid score",
            "output_is_probability": False,
            "scientific_note": (
                "This score is NOT a calibrated probability estimate. "
                "It is a class-balanced decision score produced by Platt scaling "
                "with class_weight='balanced'. The balanced weighting corrects for "
                "the 98:2 fake-to-real ratio in the calibration set, yielding a score "
                "that is useful for threshold-based decision-making but does NOT "
                "represent the true posterior probability P(fake|x) at any specific "
                "deployment prevalence."
            ) if "Balanced" in w_label else (
                "This score is the raw sigmoid output of the multimodal network."
            ),
            "thresholds": {
                "authentic": w_best["t_low"],
                "synthetic": w_best["t_high"]
            },
            "states": {
                "AUTHENTIC": f"score < {w_best['t_low']:.2f}",
                "SUSPECT": f"{w_best['t_low']:.2f} <= score <= {w_best['t_high']:.2f}",
                "SYNTHETIC": f"score > {w_best['t_high']:.2f}"
            },
            "dev_metrics": {
                "mcc": w_best["mcc"],
                "bacc": w_best["bacc"],
                "precision": w_best["precision"],
                "recall": w_best["recall"],
                "specificity": w_best["specificity"],
                "f1": w_best["f1"],
                "abstention_rate": w_best["abstention_rate"],
                "confusion_matrix_confident": w_best["cm"],
            },
            "calibration_diagnostics": {
                "brier": w_diag["brier"],
                "ece": w_diag["ece"],
            }
        }
        policy_path = os.path.join(SAVE_DIR, "V22_4F_OPERATING_POLICY.json")
        with open(policy_path, "w") as f:
            json.dump(policy, f, indent=4)
        print(f"Saved operating policy to {policy_path}")

    print("\nPHASE 6B COMPLETE.")

if __name__ == "__main__":
    main()
