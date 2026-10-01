# CODEX FINAL ACCEPTANCE AUDIT

**Date:** 2026-09-28
**Scope:** MediaDNA V22.3 End-to-End Forensic Pipeline
**Status:** Read-Only Independent Verification

## 1. FINAL PIPELINE
- **Trace Path:** Validated. The `backend/inference.py` strictly follows: Upload → `_extract_audio_melspec` / `_extract_video_frames` (Preprocessing) → Model B (Visual) & Model C (Audio) → Simple Fusion (0.5/0.5) → Platt Calibrator → Decision mapping → Schema/Report.
- **Model Checkpoints:** 
  - `V22_3B_VISUAL_CHECKPOINT.pth` (SHA-256: `B2B592CB...3C92`)
  - `V22_3C_AUDIO_CHECKPOINT.pth` (SHA-256: `CC559590...B473`)
- **Calibration Artifact:** `V22_3_PLATT_CALIBRATOR.pkl` (SHA-256: `A568BC85...A91C`)
- **Fusion Implementation:** Fixed weighting (`0.5 * visual + 0.5 * audio`), implemented directly in `backend/inference.py`.
- **Threshold/Decision Policy:** The threshold is hardcoded at `0.70`.
- **Preprocessing:** Audio uses `torchaudio.transforms.MelSpectrogram` exactly matched to training.

## 2. MODEL B / MODEL C
- **Training/Dev Splits:** Both models correctly isolated `V22_2_TRAIN.csv` for training and `V22_2_DEV.csv` for development evaluation.
- **Architecture:** Both utilize subsets of the `VideoCAVMAEFT` model (Model B = Visual Encoder + MLP, Model C = Audio Encoder + MLP). The `mean(dim=-1)` pooling flaw from V14 was recognized, and late fusion was correctly employed to bypass it.
- **Parameter Provenance:** Models were initialized from V14 checkpoints and fine-tuned independently. 
- **Verification:** I independently confirmed that raw predictions mapped correctly to the reported evaluation metrics on the DEV split without leaking into the locked test set.

## 3. TRIAD AUDIT
- **Intervention Testing:** The diagnostic triad correctly routed identical DEV samples through the repaired Late Fusion pipeline under silence and replacement conditions.
- **Score Aggregation:** The fusion scores appropriately combined visual outputs with the modified audio outputs.
- **Findings Validation:** When audio was silenced, the Fusion ROC-AUC dropped to `0.533` (identical to visual-only). When replaced, it dropped to `0.538`. 
- **Claim Support:** The evidence is *consistent with audio contribution*, proving that the pipeline possesses genuine audio sensitivity. It does NOT definitively prove causal reliance on precise phonetic manipulation, but rather observational sensitivity to audio distributions.

## 4. LATE FUSION AUDIT
- **Implementation:** The fusion uses fixed equal weighting (Option B): `0.5 * visual + 0.5 * audio`.
- **Provenance:** The weights were selected heuristically, not trained or tuned via gradient descent. No DEV, CAL, or TEST data was leaked to determine these weights. 

## 5. CALIBRATION AUDIT
- **Implementation:** Platt scaling (Logistic Regression) was successfully fitted exclusively on `v22_2_calibration.csv` (CAL split). 
- **Reload Behavior:** The calibrator successfully serializes and reloads via `pickle`. 
- **Threshold Provenance:** The decision threshold of `0.70` appears to have been selected heuristically to minimize false positives, but it resulted in a heavily skewed operating point (predicting 100% SYNTHETIC on the test set). I am explicitly **flagging** this threshold as mathematically uncalibrated for balanced accuracy.

## 6. FINAL LOCKED TEST
I independently extracted predictions from the frozen V22.3 pipeline on the 187 samples in `v22_2_test.csv` (137 Fake, 50 Real) and recomputed the metrics:
- **ROC-AUC:** 0.6013 (Reported matched)
- **PR-AUC:** 0.7988 (Reported matched)
- **Brier:** 0.2557 (Reported matched)
- **Accuracy:** 0.7326
- **Balanced Accuracy:** 0.5000
- **Precision:** 0.7326
- **Recall:** 1.0000
- **Specificity:** 0.0000 (Model predicted 100% "Fake" at 0.7 threshold)

The metrics perfectly verify the provided raw predictions. The test set was kept strictly isolated.

## 7. SCIENTIFIC CLAIM AUDIT
I scanned the repository for unsupported terminology.
- **Flagged:** `README_MEDIADNA.md` claims it is a "robust inference wrapper".
- **Flagged:** `V22_AUDIO_SHORTCUT_REPORT.md` claims "proving a causal audio shortcut".
- **Recommendations:** 
  - Replace "robust" with "functionally integrated".
  - Replace "causal proof" with "observational evidence consistent with strong reliance". 
  - Avoid claims of "production-readiness" or "high accuracy" as the LOCKED TEST ROC-AUC is only 0.601, and specificity at threshold is 0.0.

## 8. PROVENANCE AUDIT
The following items remain **UNKNOWN**:
- The exact hyperparameter grid, random seed, and loss curves for the *original* V14 AVFF checkpoint from which the V22.3 specialists were initialized.
- The exact provenance of the video compression and artifacts inside the FakeAVCeleb dataset slices. 

## 9. FINAL PRODUCT AUDIT
- **End-to-End Flow:** Verified. The `backend/main.py` successfully exposes FastAPI endpoints that run the `OpenAVFFService`. 
- **Frontend:** The frontend consumes these endpoints correctly.
- **Integrity Checks:** Implemented file-size limits (500MB), duration limits (300s), missing `ffprobe` pathing was manually fixed, and security handling wraps the core inference gracefully. Mocks have been successfully replaced by the real PyTorch models.

## 10. FINAL VERDICT

**VERIFIED WITH LIMITATIONS**

**Software Completion vs Scientific Validation:**
The **SOFTWARE COMPLETION** is fully verified. The codebase is end-to-end operational, the inference API is robustly containerized in FastAPI, the frontend connects seamlessly to real model weights, and the pipeline correctly avoids OOM errors through sequential processing. 

However, the **SCIENTIFIC VALIDATION** is limited. While the data hygiene (strict isolation of TRAIN, DEV, CAL, and LOCKED TEST) was perfectly maintained, the final model's actual predictive power on the locked test set is modest (ROC-AUC 0.601), and the chosen calibration threshold (0.7) forces a 0.0 specificity operating point.

**Suitability:**
- This project is **HIGHLY SUITABLE** to present as a "working final-year prototype". The engineering, architectural design, data isolation, UI/UX, and forensic reporting pipelines are exceptionally well-constructed. 
- In a research paper, you **MUST NOT MAKE** claims of "state-of-the-art accuracy", "robust deepfake detection", or "causal feature localization". Instead, position it as a modular, structurally sound pipeline for auditing multimodal alignment, highlighting the diagnostic triad as a novel behavioral probing methodology. 
