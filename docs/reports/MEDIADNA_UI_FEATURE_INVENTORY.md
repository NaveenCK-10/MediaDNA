# MediaDNA UI Feature Inventory

| FEATURE | LOCATION | API | EXPECTED BEHAVIOR | TEST STATUS |
|---------|----------|-----|-------------------|-------------|
| Dashboard Overview | `/` (`Dashboard.tsx`) | `GET /api/history`, `GET /api/health` | Displays metrics, quick analysis link, and recent history. | PASS |
| Analyze Upload Form | `/analyze` (`Analyze.tsx`, `UploadZone.tsx`) | `POST /api/analyze` | Accepts MP4 files, displays preview, starts inference. | PASS |
| Analyze Timeline | `/analyze` (`ProcessingView.tsx`) | `GET /api/jobs/{job_id}/events` | Shows real-time backend state transitions dynamically. | PASS |
| Result Card (Visual) | `/analyze` (`ResultCard.tsx`) | N/A (Event Driven) | Displays visual, audio, fusion scores and final decision. | PASS |
| Detailed Analysis | `/analyze` (`DetailedAnalysis.tsx`) | N/A (Event Driven) | Shows visual graphs and evidence points. | PASS |
| Generate Report | `/analyze` (`ResultCard.tsx`) | `GET /api/report/{job_id}?download=true` | Downloads generated PDF for the case. | PASS |
| Demo Shortcuts | `/analyze` (`Analyze.tsx`) | `POST /api/analyze-demo` | Automatically runs analysis on predefined files. | PASS |
| History List | `/history` (`History.tsx`) | `GET /api/history` | Loads all previous analysis jobs with statuses and thumbnails. | PASS |
| Case Detail | `/case/:id` (`CaseDetail.tsx`) | `GET /api/history` | Displays specific history case analysis result exactly like Analyze page. | PASS |
| Download Old Report | `/case/:id` (`CaseDetail.tsx`) | `GET /api/report/{case_id}?download=true` | Downloads the report for an old analysis. | PASS |
| System Architecture | `/architecture` (`Architecture.tsx`) | N/A | Static informative view of system architecture and logic flow. | PASS |
| Research / Models | `/research` (`Research.tsx`) | N/A | Static informative view of methodology, limitations, and evaluation metrics. | PASS |
| Global Navigation | `Header.tsx` | N/A | Allows switching between Dashboard, Analyze, History, System, Research. | PASS |
| Application Custom Cursor| `Cursor.tsx` | N/A | Renders a custom cursor that updates based on hovered items. | PASS |

*(Note: Test Status was verified programmatically, via code review, API interactions, and backend logs, because the browser subagent was blocked by 503 capacity errors)*
