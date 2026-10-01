# Robustness and Modality-Condition Audit

## Verdict: NOT EXECUTED / UNKNOWN

No reproducible robustness run artifact was located that ties H264/social transcode, resolution, frame rate, audio codec/noise/music, A/V offset, blur, brightness, crop, or screen-recording transforms to source code, transformed assets, predictions, and measured metrics.

The dataset itself has all four filename-level modality buckets, including 10,225 FVRA and 1,016 RVFA MP4s. V21.4’s six rows contain two RVRA, two RVFA, and two FVRA entries but no FVFA entry. Thus its aggregate metrics cannot represent all conditions and no condition-level recall is reportable.

V22.1’s ablation table reports `FVRA_Recall=0.0` only for “Model A”, but neither a generating run nor per-condition prediction file supports it. Treat it as **UNVERIFIED**, not a measured failure rate.

No latency, AUC, balanced accuracy, F1, specificity, recall, abstention, or failure rate for a robustness suite is available.
