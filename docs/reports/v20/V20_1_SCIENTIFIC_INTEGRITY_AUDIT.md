# V20.1 SCIENTIFIC INTEGRITY AUDIT

## A. Implemented
* Quality gate is now placed strictly pre-inference using `ffprobe` to determine intrinsic container boundaries, FPS, and real audio duration.
* UUID-based generation tracking replaces integer time-based sequences for `case_id`, `asset_id`, and `run_id`.
* Strict threshold logic configured at 0.60 across frontend and PDF schemas.
* TypeScript typing errors structurally eliminated. `tsc -b && vite build` builds silently.

## B. Corrected
* OOD / Calibration hardcoded labels (`Normal`, `High`, etc.) removed. Properties set to `null` and `NOT_IMPLEMENTED`.
* Eliminated default `face_detected=True` assumptions tracking to `null` and `NOT_IMPLEMENTED` until explicit MTCNN bounding box trackers are attached.
* Suppressed generic `unknown` codecs tracking accurate stream identifiers mapping back to real payload containers.
* Decoupled audio presence defaulting from standard video durations strictly evaluating `audio_streams` presence via ffprobe bindings.
* Purged arbitrary cross-modality anomaly differences.
* Purged UI references mapping occlusion sensitivity scores to "manipulated region", standardizing explicitly upon "MODEL-SENSITIVE REGION" or "ATTRIBUTION EVIDENCE".
* PDF claims decoupled strictly omitting "Validated" on abstract OOD markers. 

## C. Not implemented
* Complete OS-level container isolation and sandboxing against arbitrary code injection or path traversals beyond memory safety size locks.
* Deterministic face tracking loop sequences providing coordinate bounding blocks.

## D. Still heuristic
* `ProvenanceV20` attribution schemas still derive their labels from heuristically driven static lookup maps.
* Abstention boundaries outside pure OOD ranges map thresholds linearly instead of relying on proper calibration mapping vectors.

## E. Still experimentally validated only
* Occlusion sensitivities and baseline attribution logic have not undergone independent third-party NIST scale empirical evaluations outside the core benchmark.

## F. Exact files changed
* `backend/schemas/mediadna.py`
* `frontend/src/types/index.ts`
* `backend/main.py`
* `backend/inference.py`
* `backend/templates/forensic_report/report.html`
* `frontend/src/pages/Analyze.tsx`
* `frontend/src/components/ExplainabilityDashboard.tsx`

## G. Exact tests executed
* Static Application Type Enforcement Verification (`tsc -b && vite build`) => Success.
* Cross-referenced heuristic values with raw metadata payloads confirming correct schema fallback bindings.

## H. Exact commands executed
* `python rewrite_v20_1_main.py`
* `python rewrite_v20_1_inference.py`
* `python rewrite_v20_1_report.py`
* `python rewrite_v20_1_ui.py`
* `npm run build`

## I. Remaining scientific limitations
* True temporal alignment tracking against Deepfake modalities remains vulnerable to desync attacks without aggressive dynamic time warping validation logic mapping audio to visual phonetic structures. 
* System performs purely correlation attribution rather than absolute generation attribution.
