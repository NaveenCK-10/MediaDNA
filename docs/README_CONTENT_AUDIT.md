# README Content Audit

## Claims Checked
- AVFF dependency and origin accurately cited (Oorloff et al., CVPR 2024).
- 3-state decision policy (Authentic, Uncertain, Synthetic) strictly verified as the current operational state of MediaDNA.
- V22.4 retraining status explicitly marked as "Results Pending" as execution is currently active in the background.

## Metrics Checked
- `ROC-AUC = 0.8974` (DEV Baseline)
- `ROC-AUC = 0.9094` (Locked Test Baseline)
- Specificity = `0.0000` accurately retained and explained to demonstrate the failure mode of forced binary thresholds.
- Metrics match `V22_FINAL_REPORT.md` and historical JSON dumps.

## Repository Files Consulted
- `package.json` (for frontend stack dependencies)
- `backend/main.py` (for active API routes)
- `docs/reports/v22/V22_FINAL_REPORT.md` (for historical context and ablation results)
- `V22_4_recovery/` (for current retraining status)

## Stale Claims Removed
- Removed "64.69% accuracy" and "Critical failure on FakeVideo+RealAudio" claims from the root, as these applied to the old unimodal architecture from V20/V21, not the `VideoCAVMAEFT` AVFF multimodal core.

## Missing Information
- **Screenshots**: Currently missing from the repository. Added a note internally to include them after the UI freeze.
- **V22.4 Retraining Final Metrics**: Pending completion of `train_multimodal.py`.

## Final README Status
- **COMPLETED**: The README is correctly structured with a premium scientific layout, including Mermaid diagrams, badges, and strict forensic disclaimers.
