# MEDIADNA FINAL FREEZE REPORT (PHASE 7)

## 1. Security & Repository Status
- **NVIDIA API Credential Scope**: Hardcoded credentials removed from `backend/modules/report_generator/nvidia_narrative.py` and scratch scripts. Code defaults to deterministic fallback if environment variables are missing.
- **Git History Scan**: Full repository history scanned for API keys, bearer tokens, passwords, and `.env` files.
- **Current-source security status**: No exposed credential was found in the current staged source. A historical NVIDIA credential exposure remains in Git history and requires credential revocation/rotation.

## 2. Frozen Scientific Artifacts
The following artifacts have been generated, frozen, and verified for internal consistency:
- `V22_4F_MULTIMODAL_StageB_Ep2.pth` (Primary checkpoint)
- `V22_4F_DEV_raw_scores.npz`
- `V22_4F_CAL_raw_scores.npz`
- `V22_4F_LOCKED_TEST_SCORES.npz`
- `V22_4F_LOCKED_TEST_FINAL.json`
- `V22_4F_OPERATING_POLICY.json`

Archived Phase-6B calibration experiment artifacts (NOT used by the frozen V22.4F inference path):
- `V22_4F_PLATT_CALIBRATOR.pkl`
- `V22_4F_PLATT_CALIBRATOR_A_unweighted.pkl`
- `V22_4F_PLATT_CALIBRATOR_B_balanced.pkl`

> [!IMPORTANT]
> The scientific experimentation is permanently locked. No further modifications, retraining, or threshold tuning will occur.

## 3. Production Model Selection & Frozen Policy
- **Selected Model**: `VideoCAVMAEFT` (V22.4F Multimodal Checkpoint).
- **Score Output**: Raw Sigmoid Decision Score (NOT calibrated probability).
- **Frozen 3-State Policy**:
  - `Decision Score < 0.20` → **AUTHENTIC**
  - `0.20 ≤ Decision Score ≤ 0.45` → **UNCERTAIN**
  - `Decision Score > 0.45` → **SYNTHETIC**

## 4. Final Exact Metrics (Locked Test)
On the final immutable 2,219-sample locked test set:
- **Total Samples Evaluated**: 2,219
- **Abstention Rate (Uncertain Zone)**: 4.9%
- **ROC-AUC**: 0.7500
- **MCC** (confident): 0.1679
- **Specificity** (confident): 0.8750
- **Recall** (confident): 0.6621
- **Precision** (confident): 0.9956
- **Balanced Accuracy** (confident): 0.7686

## 5. Application Test Matrix (Runtime Audit)
The end-to-end processing pipeline was verified successfully on the runtime architecture:
- **Valid Video Check**: PASS
- **Valid Video with Audio**: PASS
- **No-Audio Video**: PASS
- **Corrupt / Unsupported Format Rejection**: PASS (Returns HTTP 400 with graceful JSON response)
- **Oversized / Long-duration Constraints**: PASS
- **SSE Processing State Streaming**: PASS (Proper status stream emitted from QUEUED to COMPLETED)
- **JSON Result Rendering**: PASS
- **History Retrieval**: PASS

*Terminology Audit Completed:* Terminology across the repository (e.g., schemas, reports, frontend UI) has been audited and updated to use scientifically supported phrases like "decision score", "model-sensitive region", and "supported by available evidence", eliminating legacy unsupported terms like "fake probability" or "exact generator identified".

## 6. Known Non-Blocking Limitations
- **FakeVideo-RealAudio Blind Spot**: The model suffers from severe underperformance (23% recall) when evaluating visually manipulated deepfakes paired with original/authentic audio tracks. The cross-modal attention layers rely heavily on audio-visual desynchronization.
- **Low MCC Context**: The final MCC (0.1679) reflects better-than-chance discrimination but limits the model's utility as a high-confidence autonomous tool. It functions best as an evidence-gathering aid rather than a definitive authority.
- **Score is Not Calibrated Probability**: Outputs represent non-calibrated decision scores ordinal to the selected thresholds.

---
**STATUS:** MediaDNA is frozen for final paper, demonstration, and viva preparation. ALL MODEL DEVELOPMENT HAS STOPPED.
