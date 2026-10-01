"""
V22.4F PHASE 7 — FINAL LOCKED TEST
====================================
ONE-SHOT evaluation. Do NOT rerun for tuning.

FROZEN MODEL:    V22_4F_MULTIMODAL_StageB_Ep2.pth
FROZEN SCORE:    Raw sigmoid (decision score, NOT calibrated probability)
FROZEN POLICY:
    score < 0.20            -> AUTHENTIC
    0.20 <= score <= 0.45   -> UNCERTAIN
    score > 0.45            -> SYNTHETIC
"""

import os, sys, csv, time, json, hashlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score
)
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

# FROZEN thresholds — do NOT modify
T_LOW  = 0.20
T_HIGH = 0.45

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as f:
            self.rows = list(csv.DictReader(f))
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.labels = []
        self.types = []
        self.paths = []
        for r in self.rows:
            self.labels.append(0 if r["type"] == "RealVideo-RealAudio" else 1)
            self.types.append(r["type"])
            fname = r["path"]
            dir_col = r.get("", "") or [v for k,v in r.items() if k is None or k == ""][0]
            if isinstance(dir_col, list): dir_col = dir_col[0]
            self.paths.append(os.path.join(DATASET_DIR, dir_col.replace("FakeAVCeleb/", ""), fname))

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

def main():
    print("=" * 60)
    print("V22.4F PHASE 7 — FINAL LOCKED TEST (ONE-SHOT)")
    print("=" * 60)
    print(f"Frozen Policy: AUTHENTIC < {T_LOW}, UNCERTAIN [{T_LOW}, {T_HIGH}], SYNTHETIC > {T_HIGH}")
    print(f"Score Type: Raw sigmoid decision score (NOT calibrated probability)")

    # ── Checkpoint hash ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(SAVE_DIR, "V22_4F_MULTIMODAL_StageB_Ep2.pth")
    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    ckpt_hash = sha256.hexdigest()
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"SHA-256: {ckpt_hash}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = VideoCAVMAEFT()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.to(DEVICE)
    model.eval()
    print("Model loaded and set to eval mode.")

    # ── Load locked test ─────────────────────────────────────────────────────
    test_csv = os.path.join(PROJECT_ROOT, "data", "v22_2_test_locked.csv")
    test_dataset = MultimodalDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    n_total = len(test_dataset)
    gt_real = sum(1 for l in test_dataset.labels if l == 0)
    gt_fake = sum(1 for l in test_dataset.labels if l == 1)
    print(f"\nLocked Test: {n_total} samples ({gt_real} real, {gt_fake} fake)")

    # ── Inference ────────────────────────────────────────────────────────────
    all_scores = []
    all_labels = []
    all_paths = []
    t0 = time.time()

    with torch.no_grad():
        for i, (a, v, l, paths) in enumerate(test_loader):
            if i % 200 == 0: print(f"  Progress: {i}/{len(test_loader)} batches")
            a, v = a.to(DEVICE), v.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(a, v)
            prob = torch.sigmoid(out)[:, 0].cpu().float().numpy()
            all_scores.extend(prob)
            all_labels.extend(l.numpy()[:, 0])
            all_paths.extend(paths)

    inference_dur = time.time() - t0
    scores = np.array(all_scores, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.float64).astype(int)
    print(f"  Inference complete: {inference_dur:.1f}s")

    # ── Save raw scores ──────────────────────────────────────────────────────
    scores_path = os.path.join(SAVE_DIR, "V22_4F_LOCKED_TEST_SCORES.npz")
    np.savez(scores_path, scores=scores, labels=labels, paths=all_paths)
    print(f"  Saved scores: {scores_path}")

    # ── Score distribution ───────────────────────────────────────────────────
    print(f"\n--- Raw Score Distribution ---")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")

    real_mask = labels == 0
    fake_mask = labels == 1
    print(f"  Real (n={real_mask.sum()}): mean={scores[real_mask].mean():.4f}, std={scores[real_mask].std():.4f}")
    print(f"  Fake (n={fake_mask.sum()}): mean={scores[fake_mask].mean():.4f}, std={scores[fake_mask].std():.4f}")

    # ── 3-state predictions ─────────────────────────────────────────────────
    pred_auth = scores < T_LOW
    pred_uncertain = (scores >= T_LOW) & (scores <= T_HIGH)
    pred_synth = scores > T_HIGH

    n_auth = pred_auth.sum()
    n_uncertain = pred_uncertain.sum()
    n_synth = pred_synth.sum()
    abstention_rate = n_uncertain / n_total

    print(f"\n--- 3-State Predictions ---")
    print(f"  AUTHENTIC:  {n_auth}")
    print(f"  UNCERTAIN:  {n_uncertain}")
    print(f"  SYNTHETIC:  {n_synth}")
    print(f"  Abstention: {abstention_rate*100:.1f}%")

    # ── Binary metrics (confident only) ──────────────────────────────────────
    confident_mask = pred_auth | pred_synth
    y_conf = labels[confident_mask]
    p_conf = pred_synth[confident_mask].astype(int)

    cm_conf = confusion_matrix(y_conf, p_conf)
    tn, fp, fn, tp = cm_conf.ravel() if cm_conf.shape == (2,2) else (0,0,0,0)

    mcc_conf = matthews_corrcoef(y_conf, p_conf)
    bacc_conf = balanced_accuracy_score(y_conf, p_conf)
    prec_conf = precision_score(y_conf, p_conf, zero_division=0)
    rec_conf = recall_score(y_conf, p_conf, zero_division=0)
    spec_conf = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_conf = f1_score(y_conf, p_conf, zero_division=0)

    # ── Full binary metrics (threshold=0.5) ──────────────────────────────────
    p_bin_full = (scores >= 0.5).astype(int)
    auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    mcc_full = matthews_corrcoef(labels, p_bin_full)
    bacc_full = balanced_accuracy_score(labels, p_bin_full)
    cm_full = confusion_matrix(labels, p_bin_full)

    # ── Per-class recall ─────────────────────────────────────────────────────
    # Among confident predictions
    per_class = {}
    for t in set(test_dataset.types):
        idx = [i for i, tp in enumerate(test_dataset.types) if tp == t]
        t_scores = scores[idx]
        t_labels = labels[idx]
        t_auth = (t_scores < T_LOW).sum()
        t_unc = ((t_scores >= T_LOW) & (t_scores <= T_HIGH)).sum()
        t_synth = (t_scores > T_HIGH).sum()
        is_real = 1 if t == "RealVideo-RealAudio" else 0
        if is_real:
            correct = t_auth
        else:
            correct = t_synth
        recall_cls = correct / len(idx) if len(idx) > 0 else 0
        per_class[t] = {
            "n": len(idx),
            "auth": int(t_auth), "uncertain": int(t_unc), "synth": int(t_synth),
            "recall": round(float(recall_cls), 4),
            "mean_score": round(float(t_scores.mean()), 4),
        }

    # ── Error analysis ───────────────────────────────────────────────────────
    # False positives: real samples predicted SYNTHETIC
    fp_indices = [i for i in range(n_total) if labels[i] == 0 and scores[i] > T_HIGH]
    # False negatives (confident): fake samples predicted AUTHENTIC
    fn_indices = [i for i in range(n_total) if labels[i] == 1 and scores[i] < T_LOW]

    fp_files = [os.path.basename(all_paths[i]) for i in fp_indices]
    fn_files = [os.path.basename(all_paths[i]) for i in fn_indices]

    # ── Print everything ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LOCKED TEST RESULTS")
    print(f"{'='*60}")
    print(f"\n1.  Total samples:          {n_total}")
    print(f"2.  Ground-truth:           {gt_real} real, {gt_fake} fake")
    print(f"3.  Score distribution:     min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, median={np.median(scores):.4f}")
    print(f"4.  AUTHENTIC:              {n_auth}")
    print(f"    UNCERTAIN:              {n_uncertain}")
    print(f"    SYNTHETIC:              {n_synth}")
    print(f"5.  Abstention rate:        {abstention_rate*100:.1f}%")
    print(f"6.  Confusion matrix (confident):")
    print(f"    {cm_conf}")
    print(f"7.  ROC-AUC:                {auc:.4f}")
    print(f"8.  PR-AUC:                 {pr_auc:.4f}")
    print(f"9.  Balanced Accuracy:      {bacc_conf:.4f} (confident) / {bacc_full:.4f} (full)")
    print(f"10. MCC:                    {mcc_conf:.4f} (confident) / {mcc_full:.4f} (full)")
    print(f"11. Precision:              {prec_conf:.4f}")
    print(f"12. Recall:                 {rec_conf:.4f}")
    print(f"13. Specificity:            {spec_conf:.4f}")
    print(f"14. F1:                     {f1_conf:.4f}")
    print(f"15. False Positives ({len(fp_files)} real->SYNTH): {fp_files}")
    print(f"    False Negatives ({len(fn_files)} fake->AUTH):  {fn_files[:20]}{'...' if len(fn_files) > 20 else ''}")
    print(f"16. Per-class recall:")
    for cls, info in sorted(per_class.items()):
        print(f"    {cls:<30s}: n={info['n']:>5}, recall={info['recall']:.4f}, "
              f"mean_score={info['mean_score']:.4f}, "
              f"auth={info['auth']}, unc={info['uncertain']}, synth={info['synth']}")
    print(f"17. Checkpoint SHA-256:     {ckpt_hash}")
    print(f"18. Policy version:         V22_4F_RAW_SIGMOID_v1 (t_low={T_LOW}, t_high={T_HIGH})")

    # ── Save immutable JSON ──────────────────────────────────────────────────
    result = {
        "phase": "PHASE 7 — LOCKED TEST",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": "V22_4F_MULTIMODAL_StageB_Ep2.pth",
        "checkpoint_sha256": ckpt_hash,
        "score_type": "raw sigmoid decision score (NOT calibrated probability)",
        "policy_version": "V22_4F_RAW_SIGMOID_v1",
        "thresholds": {"authentic": T_LOW, "uncertain_low": T_LOW, "uncertain_high": T_HIGH, "synthetic": T_HIGH},
        "total_samples": int(n_total),
        "ground_truth": {"real": int(gt_real), "fake": int(gt_fake)},
        "score_distribution": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
            "median": round(float(np.median(scores)), 4),
        },
        "three_state_counts": {
            "AUTHENTIC": int(n_auth),
            "UNCERTAIN": int(n_uncertain),
            "SYNTHETIC": int(n_synth),
        },
        "abstention_rate": round(float(abstention_rate), 4),
        "metrics_confident": {
            "ROC-AUC": round(float(auc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "Balanced_Accuracy": round(float(bacc_conf), 4),
            "MCC": round(float(mcc_conf), 4),
            "Precision": round(float(prec_conf), 4),
            "Recall": round(float(rec_conf), 4),
            "Specificity": round(float(spec_conf), 4),
            "F1": round(float(f1_conf), 4),
            "Confusion_Matrix": cm_conf.tolist(),
        },
        "metrics_full_binary": {
            "Balanced_Accuracy": round(float(bacc_full), 4),
            "MCC": round(float(mcc_full), 4),
            "Confusion_Matrix": cm_full.tolist(),
        },
        "per_class_recall": per_class,
        "errors": {
            "false_positives_real_to_synth": fp_files,
            "false_negatives_fake_to_auth_count": len(fn_files),
            "false_negatives_fake_to_auth_sample": fn_files[:50],
        },
        "inference_duration_sec": round(float(inference_dur), 1),
        "IMMUTABLE": True,
        "DO_NOT_RETUNE": True,
    }

    result_path = os.path.join(SAVE_DIR, "V22_4F_LOCKED_TEST_FINAL.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved: {result_path}")
    print("\nPHASE 7 COMPLETE. DO NOT RERUN.")

if __name__ == "__main__":
    main()
