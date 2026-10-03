import asyncio
from typing import Dict, Any, List
import json
from datetime import datetime, timezone

STAGES = [
    "QUEUED",
    "UPLOADING",
    "VALIDATING",
    "INSPECTING_MEDIA",
    "EXTRACTING_VIDEO",
    "VISUAL_ANALYSIS",
    "EXTRACTING_AUDIO",
    "AUDIO_PREPROCESSING",
    "AUDIO_ANALYSIS",
    "LATE_FUSION",
    "CALIBRATION",
    "DECISION",
    "FORENSIC_EVIDENCE",
    "REPORT_GENERATION",
    "COMPLETED",
    "FAILED"
]

STAGE_LABELS = {
    "QUEUED": "Queued",
    "UPLOADING": "Uploading media",
    "VALIDATING": "Validating file format",
    "INSPECTING_MEDIA": "Inspecting media streams",
    "EXTRACTING_VIDEO": "Extracting video frames",
    "VISUAL_ANALYSIS": "Visual Analysis",
    "EXTRACTING_AUDIO": "Extracting audio track",
    "AUDIO_PREPROCESSING": "Audio Preprocessing",
    "AUDIO_ANALYSIS": "Audio Analysis",
    "LATE_FUSION": "Fusing modalities",
    "CALIBRATION": "Extracting decision score",
    "DECISION": "Applying decision policy",
    "FORENSIC_EVIDENCE": "Aggregating evidence",
    "REPORT_GENERATION": "Generating PDF report",
    "COMPLETED": "Completed",
    "FAILED": "Failed"
}

def get_iso_time():
    return datetime.now(timezone.utc).isoformat()

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queues: Dict[str, asyncio.Queue] = {}

    def create_job(self, job_id: str):
        now = get_iso_time()
        self.jobs[job_id] = {
            "analysis_id": job_id,
            "status": "QUEUED",
            "stage": "QUEUED",
            "stage_index": 0,
            "stage_total": len(STAGES),
            "stage_label": STAGE_LABELS["QUEUED"],
            "message": "Waiting for analysis to start...",
            "progress": None,
            "started_at": now,
            "updated_at": now,
            "stage_started_at": now,
            "stage_completed_at": None,
            "completed_stages": [],
            "events": [{
                "timestamp": now,
                "stage": "QUEUED",
                "status": "started",
                "message": "Job created"
            }],
            "result": None,
            "error_code": None
        }
        self.queues[job_id] = asyncio.Queue()

    async def _emit(self, job_id: str):
        if job_id in self.queues:
            await self.queues[job_id].put(json.dumps(self.jobs[job_id]))

    async def update_stage_status(self, job_id: str, stage: str, status: str, message: str, progress: Any = None):
        if job_id not in self.jobs:
            return
            
        job = self.jobs[job_id]
        now = get_iso_time()
        
        if status == "started":
            job["stage"] = stage
            job["stage_index"] = STAGES.index(stage) if stage in STAGES else -1
            job["stage_label"] = STAGE_LABELS.get(stage, stage)
            job["status"] = "PROCESSING"
            job["stage_started_at"] = now
            job["stage_completed_at"] = None
            
        if status == "completed":
            job["stage_completed_at"] = now
            if stage not in job["completed_stages"]:
                job["completed_stages"].append(stage)
            
        job["message"] = message
        job["progress"] = progress
        job["updated_at"] = now
        
        event = {
            "timestamp": now,
            "stage": stage,
            "status": status,
            "message": message
        }
        job["events"].append(event)
        
        await self._emit(job_id)

    async def finish_job(self, job_id: str, result: dict):
        if job_id not in self.jobs:
            return
        now = get_iso_time()
        job = self.jobs[job_id]
        job["status"] = "COMPLETED"
        job["stage"] = "COMPLETED"
        job["stage_index"] = STAGES.index("COMPLETED")
        job["stage_label"] = STAGE_LABELS["COMPLETED"]
        job["message"] = "Forensic analysis complete"
        job["progress"] = None
        job["updated_at"] = now
        job["stage_completed_at"] = now
        job["result"] = result
        
        job["events"].append({
            "timestamp": now,
            "stage": "COMPLETED",
            "status": "completed",
            "message": "Analysis completed successfully"
        })
        
        await self._emit(job_id)
        if job_id in self.queues:
            await self.queues[job_id].put(None)

    async def fail_job(self, job_id: str, message: str, error_code: str):
        if job_id not in self.jobs:
            return
        now = get_iso_time()
        job = self.jobs[job_id]
        job["status"] = "FAILED"
        job["stage"] = "FAILED"
        job["stage_index"] = STAGES.index("FAILED")
        job["stage_label"] = STAGE_LABELS["FAILED"]
        job["message"] = message
        job["progress"] = None
        job["updated_at"] = now
        job["stage_completed_at"] = now
        job["error_code"] = error_code
        
        job["events"].append({
            "timestamp": now,
            "stage": "FAILED",
            "status": "failed",
            "message": message
        })
        
        await self._emit(job_id)
        if job_id in self.queues:
            await self.queues[job_id].put(None)

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

job_manager = JobManager()
