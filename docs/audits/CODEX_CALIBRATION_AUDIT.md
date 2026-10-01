# Calibration and Uncertainty Audit

## Verdict: NOT VALIDATED

The backend’s trust object returns `calibrated_probability=None`, `calibration_status="NOT_VALIDATED"`, `ood_status="NOT_IMPLEMENTED"`, and a heuristic `BORDERLINE` abstention state. The optional `calibrator.pkl` is loaded if present but its provenance, calibration split, score source, fit method, and held-out ECE/Brier results are not recorded. Its existence cannot establish calibration.

No dedicated calibration manifest was found. The V21.4 threshold is hard-coded at 0.60; its selection procedure is unknown. No ECE, Brier, reliability curve, abstention coverage/risk curve, or OOD validation result is available.

Required: a frozen calibration split disjoint by identity/source/derivative from train/dev/test; fit exactly once after candidate selection; preserve raw logits and calibrated outputs; report ECE and Brier on untouched test; make abstention state a downstream decision gate. Metadata/quality rules must remain labelled as quality checks rather than true OOD detection.
