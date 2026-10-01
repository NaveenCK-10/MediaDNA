# Current Architecture Audit

## Active path

`FastAPI /api/analyze` → temporary upload → `OpenAVFFService.analyze_video` → FFmpeg audio decode + Decord video decode → `VideoCAVMAEFT` checkpoint → auxiliary visual/audio/temporal modules → optional serialized fusion/calibrator objects → schema/history/report endpoints.

The active API loads `checkpoints/v14_fullscale/models/best_audio_model.pth`. The base model is `src/models/video_cav_mae.py:VideoCAVMAEFT`; audio, visual, and cross-modal fusion modules reside under `src/models`. The backend exposes health, model-info, analysis, job-event, report, and history routes. Frontend code calls the backend through `frontend/src/api/client.ts`.

## Findings

- V21.4 does perform a real forward pass in its benchmark code, but it uses a six-sample manifest and an unproven training provenance.
- V22.1 is not an experiment candidate. Its training code uses locked test data; Model E and Model G construct all-zero surrogate features; Models F1/F2/G are not called; output status remains `PENDING_EXECUTION`.
- Backend import was read-only validated successfully. The configured FFmpeg executable exists; `ffprobe` is not on PATH even though `_inspect_media` invokes it by name, so quality/provenance metadata can fail by environment.
- `backend/inference.py` loads state dicts with `strict=False` and reports `checkpoint_hash=None`/`UNAVAILABLE` in its returned profile. The API therefore does not yet provide a complete checkpoint audit trail.
- Quality, calibration, OOD, and abstention surface as fields, but source labels include `NOT_VALIDATED`, `NOT_IMPLEMENTED`, and `HEURISTIC`; they are not evidence of validated controls.

## Scope and time sensitivity

The working tree already contained active uncommitted backend/frontend changes. Findings involving those files were read on 2026-09-24 and must be re-read after Antigravity’s pass before implementation acceptance. This report does not modify implementation files.
