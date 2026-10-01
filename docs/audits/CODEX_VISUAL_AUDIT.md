# Visual Branch Audit

## Verdict: INCOMPLETE; no visual-only benchmark is valid

The visual preprocessing used by V21.4/V22.1 uniformly samples 16 frames, resizes directly to 224×224, and applies ImageNet normalization. It does not perform face detection, face selection, crop, alignment, multi-face policy, or face-quality gating. The live profile explicitly records face detection as `NOT_IMPLEMENTED`.

The V22.1 “visual-only” wrapper zeros audio and calls the frozen AV model, but it then trains and evaluates on the same locked six samples. It is not a valid visual benchmark. The V22.1 multitask and A/V-sync classes do not expose backbone features: both replace features with newly created zero tensors. Their outputs cannot measure visual evidence.

The current code permits a genuine forward pass, but no valid visual-only result was executed in this audit. Visual weakness may be a pipeline/experimental-design defect, a model limitation, or both; the available evidence cannot discriminate them.

Required experiment: establish identity/source-disjoint train/dev/calibration/test manifests; log decode/face/crop failures; train or freeze a documented visual specialist on train only; select thresholds on dev only; evaluate full AV, visual-only, audio-only, and FVRA/RVFA on locked test once.
