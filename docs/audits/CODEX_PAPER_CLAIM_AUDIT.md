# Claim Audit

## Unsupported or overstated claims found

- `v21_master_eval.py` writes a 52.5% “evaluated” baseline plus ablation, robustness, cross-dataset, and fusion conclusions without evaluation computation. These claims are invalid.
- `v21_1_master_reconstruction.py` writes “completely disjoint” and fixed metrics without supporting manifests or inference. These claims are invalid.
- V14/V16 reports repeat AUC 0.9145, but the named training/validation CSVs are absent. The metric must be labelled historical/unverified, not independently reproduced.
- `V21_4_BASELINE_RESULTS.json` contains real-code output according to its harness, but only six samples and unavailable train provenance. It must not support a general performance or production claim.
- V22.1 report language correctly keeps the current baseline, but its ablation table’s Model A metrics and FVRA recall lack a traceable execution artifact; the table must not be used as empirical evidence.
- Backend UI/API must not represent raw logits as calibrated authenticity probabilities, model-sensitive occlusion as manipulated-region localization, or heuristic provenance as generator attribution. Current schemas partly label these limits, but report/UI review remains needed after concurrent frontend changes settle.

No files were rewritten under the read-only audit restriction. The correct present wording is: “research prototype; performance, calibration, OOD, localization, provenance attribution, and generalization not independently validated.”
