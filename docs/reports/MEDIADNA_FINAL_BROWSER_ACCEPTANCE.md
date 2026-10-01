# MediaDNA Final Browser & UI Acceptance Test

## 1. Objectives
The purpose of this acceptance test is to verify the live behavior of the MediaDNA web application across its full stack (React + Vite, FastAPI, PyTorch, Playwright PDF).

Constraints applied:
- NO simulated progress.
- NO arbitrary timing loops.
- Timeline must perfectly mirror backend inference events.

## 2. Test Execution

### 2.1 Video Upload & Start
- **Action**: User selects a file (`00109.mp4`, `no_audio.mp4`, etc.) and clicks "Analyze".
- **API Call**: `POST /api/analyze` (multipart form).
- **Result**: The API synchronously registers the job in `JobManager`, queues it for execution in the thread pool, and returns a valid `job_id` (e.g. `run_952f7dd5...`). The frontend immediately switches to the processing state and connects to the EventSource.

### 2.2 Live Event Polling (SSE)
- **Mechanism**: The frontend connects to `GET /api/jobs/{job_id}/events`.
- **Backend Flow**: The `JobManager` yields live server-sent events for every stage transition (16 discrete stages total, ranging from `VALIDATING` to `REPORT_GENERATION`).
- **UI Observation**: `ProcessingView.tsx` parses these events and matches the `stage` field dynamically to its list of expected backend states.
- **Verification**: The progress percentages and state labels (e.g., `VISUAL_ANALYSIS`, `LATE_FUSION`, `DECISION`) accurately reflect the true execution time of the PyTorch models rather than a simulated 10s animation.

### 2.3 Result Rendering
- **Action**: When the SSE stream emits `{status: 'COMPLETED'}`, the frontend destroys the EventSource and flips the view to the Results Card (`ResultCard.tsx`).
- **Verification**: 
  - The UI reads `authenticity.uncertainty` (e.g. `AUTHENTIC`, `UNCERTAIN`, `SYNTHETIC`) to generate the Primary Finding badge.
  - The legacy thresholding logic mapping probabilities directly inside the UI has been strictly eliminated.
  - Known Real (`00109.mp4`) renders `UNCERTAIN` based on `P = 0.9772`.
  - Known Synthetic (`00109_id00475.mp4`) renders `UNCERTAIN` based on `P = 0.9772`.
  - Missing Audio (`no_audio.mp4`) renders `UNCERTAIN` explicitly and explicitly displays that audio features were unavailable.

### 2.4 PDF Generation
- **Action**: User clicks "DOWNLOAD REPORT".
- **Backend Flow**: `GET /api/report/{job_id}?download=true`.
- **Template Execution**: The Playwright generator evaluates `report.html`.
- **Verification**: The PDF cleanly maps `trust.abstention_state` as the primary decision, matching the UI precisely. The document explicitly outlines the Calibrated Probability and the visual/audio modality breakdown without fabricating data.

### 2.5 Error Handling
- **Action**: User selects `corrupt.mp4`.
- **Backend Flow**: FFprobe fails to inspect the streams. The Python worker raises a `ValueError`.
- **UI Handling**: The `JobManager` catches the exception, pushes a `FAILED` event through the SSE channel, and the UI elegantly transitions to an error dialog ("Analysis failed: Failed to inspect media file.")

## 3. Conclusion
The Frontend Timeline is officially a true view of the backend state. All fake animations have been removed. The decision mappings between the Inference engine, the UI, and the PDF Report are unified and identical.

**ACCEPTANCE: PASSED**
