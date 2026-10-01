# V21.1 REPOSITORY INVENTORY

| File/Directory | Purpose | Used by Runtime? | Status | Notes |
|---|---|---|---|---|
| `backend/main.py` | FastAPI server entrypoint | YES | VERIFIED-BY-CODE | Handles streaming upload, UUID mapping, SHA-256. |
| `backend/inference.py` | Core model execution | YES | VERIFIED-BY-CODE | VideoCAVMAEFT inference and MediaDNAProfile serialization. |
| `frontend/src/` | React client | YES | VERIFIED-BY-CODE | Types strictly mapped to backend schema. |
| `FakeAVCeleb_v1.2/` | Dataset directory | NO | UNKNOWN | Used by scripts, but test sets overlapping. |
| `final_baseline_eval.py` | Eval script | NO | VERIFIED-BY-CODE | Excludes training sets, calculates metrics. |
| `v15_*.py`, `v16_*.py` | Legacy research scripts | NO | DOCUMENT-ONLY | Deprecated experiments. |
