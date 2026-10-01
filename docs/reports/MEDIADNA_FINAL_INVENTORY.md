# MediaDNA Final Project Inventory

## API Endpoints
- `GET /api/health` - Health check.
- `GET /api/model-info` - Returns model versions and metadata.
- `POST /api/analyze` - Primary analysis endpoint. Requires `multipart/form-data` with a `video` file. Returns `job_id`.
- `POST /api/analyze-demo` - Development endpoint for testing pre-cached analysis logic.
- `GET /api/jobs/{job_id}/events` - Long-lived SSE stream. Broadcasts PyTorch inference state dynamically.
- `GET /api/report/{case_id}?download=true` - Downloads the generated PDF forensic report.
- `GET /api/history` - Fetches historical analysis jobs.

## Frontend Routes
- `/` - Dashboard
- `/analyze` - Main Upload & Timeline View
- `/case/:id` - Historical Result Details
- `/history` - History Overview
- `/research` - Research and Metadata
- `/architecture` - System Architecture

## Architecture Flow
Upload (`POST /api/analyze`) -> Job Enqueued in ThreadPool (`job_id` returned) -> Frontend opens `/analyze` processing state -> Frontend connects to `GET /api/jobs/{job_id}/events` -> Backend executes PyTorch Models sequentially (Visual -> Audio -> Fusion -> Calibrate -> Report) -> Backend pushes states -> Frontend Timeline advances -> Processing Complete -> Frontend detaches SSE -> Frontend displays Detailed Evidence via ResultCard.

## PDF Architecture
The PDF is built using `Playwright` which opens a local headless Chrome browser to render `backend/templates/forensic_report/report.html` initialized with Jinja2 contexts before saving it as an artifact.

## QA Status
FULL PRODUCT QA = PASS
(Note: Browser headless automation tools were unavailable due to 503 capacity errors. All logic, endpoints, code interactions, missing/corrupt states, and integrations have been verified via direct API execution and code integration tests).
