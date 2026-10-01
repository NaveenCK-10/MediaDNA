"""
MediaDNA FastAPI Backend.

Serves the OpenAVFF deepfake detection model through a clean REST API.
Model is loaded ONCE at startup and kept in GPU memory.
"""
import os
import sys
import time
import shutil
import logging
import tempfile
import traceback
import asyncio
import json
import uuid
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import OpenAVFFService
from backend.legacy_schemas import HealthResponse, ModelInfoResponse, ErrorResponse
from backend.schemas.mediadna import MediaDNAProfile
from backend.modules import analyze_visual_signals, extract_metadata, calculate_fusion
from backend.modules.report_generator.generator import ReportGenerator
from backend.jobs import job_manager

# ─── Configuration ───────────────────────────────────────────────────────────

# Default to the V14 fullscale checkpoint (Validated in V15.3)
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth"
)
# Fallback to medium checkpoint if local doesn't exist
FALLBACK_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth"
)

CHECKPOINT = os.environ.get("MEDIADNA_CHECKPOINT", DEFAULT_CHECKPOINT)
if not os.path.exists(CHECKPOINT):
    CHECKPOINT = FALLBACK_CHECKPOINT

# Upload directory for temporary storage
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# History file
HISTORY_FILE = os.path.join(PROJECT_ROOT, "backend", "history.json")
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# Allowed video extensions and max file size
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mediadna.api")

# ─── Global model service ────────────────────────────────────────────────────

service: OpenAVFFService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global service
    logger.info("=" * 60)
    logger.info("MediaDNA Backend Starting")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {CHECKPOINT}")

    try:
        service = OpenAVFFService(checkpoint_path=CHECKPOINT)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise

    yield

    # Cleanup
    logger.info("MediaDNA Backend shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MediaDNA API",
    description="AI-Powered Audio-Visual Deepfake Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if the backend and model are ready."""
    import torch
    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

    return HealthResponse(
        status="ok" if service and service.is_loaded else "error",
        model_loaded=service is not None and service.is_loaded,
        device=str(service.device) if service else "unknown",
        checkpoint=os.path.basename(CHECKPOINT),
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model architecture and configuration details."""
    if not service or not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="OpenAVFF",
        architecture="VideoCAVMAEFT",
        n_classes=2,
        input_visual=f"224x224",
        input_audio=f"1024x128 mel filterbank",
        num_frames=8,
        audio_sample_rate=16000,
        target_length=1024,
        num_mel_bins=128,
        dataset_mean=[0.485, 0.456, 0.406],
        dataset_std=[0.229, 0.224, 0.225],
        checkpoint=os.path.basename(CHECKPOINT),
        device=str(service.device),
        total_parameters=0,
    )


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze a video for deepfake detection.
    
    Accepts a multipart video file upload. The video is saved temporarily,
    processed through the OpenAVFF pipeline, and the result returned.
    Temporary files are cleaned up after inference.
    """
    if not service or not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for startup.")

    # ── Validate file extension ──
    filename = video.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{ext}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Validate content type ──
    content_type = video.content_type or ""
    if content_type and not content_type.startswith("video/"):
        # Allow application/octet-stream as some browsers send this
        if content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: '{content_type}'. Expected a video file.",
            )

    # ── Save uploaded file to temp location ──
    temp_path = None
    try:
        # Create temp file with original extension
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, dir=UPLOAD_DIR)
        os.close(temp_fd)

        # Stream upload to disk
        file_size = 0
        sha256_hash = hashlib.sha256()
        with open(temp_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                sha256_hash.update(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB",
                    )
                f.write(chunk)
        
        asset_hash = sha256_hash.hexdigest()

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        logger.info(f"Received video: {filename} ({file_size / (1024*1024):.1f} MB)")
        
        job_id = "run_" + uuid.uuid4().hex
        asset_id = "asset_" + asset_hash[:16]
        case_id = "case_" + asset_hash[:16] # One asset = one case for now
        
        job_manager.create_job(job_id)
        
        async def run_analysis_job(jid, t_path, fname):
            loop = asyncio.get_running_loop()
            def progress_cb(stage, status, msg, progress=None):
                asyncio.run_coroutine_threadsafe(
                    job_manager.update_stage_status(jid, stage, status, msg, progress),
                    loop
                )
                
            try:
                context = {
                    "case_id": case_id,
                    "asset_id": asset_id,
                    "run_id": jid,
                    "asset_hash": asset_hash,
                    "file_size_bytes": file_size,
                    "ingestion_timestamp": time.time()
                }
                # Run heavy inference in threadpool
                result = await loop.run_in_executor(None, lambda: service.analyze_video(t_path, progress_cb, context))
                
                # Save History
                try:
                    with open(HISTORY_FILE, "r") as f:
                        history = json.load(f)
                    history_item = result.copy()
                    history_item["id"] = jid
                    history_item["timestamp"] = time.time()
                    history_item["filename"] = fname
                    history.insert(0, history_item)
                    history = history[:50]
                    with open(HISTORY_FILE, "w") as f:
                        json.dump(history, f)
                except Exception as e:
                    logger.warning(f"Failed to save history: {e}")

                # --- REPORT GENERATION ---
                def progress_cb(stage, status, msg, progress=None):
                    asyncio.run_coroutine_threadsafe(
                        job_manager.update_stage_status(jid, stage, status, msg, progress),
                        loop
                    )
                progress_cb("REPORT_GENERATION", "started", "Generating Forensic PDF Report")
                try:
                    out_dir = os.path.join(PROJECT_ROOT, "experiments", "final_demo", "reports")
                    os.makedirs(out_dir, exist_ok=True)
                    pdf_path = os.path.join(out_dir, f"MediaDNA_Forensic_Report_{jid}.pdf")
                    
                    generator = ReportGenerator()
                    # It's blocking so we run it in executor
                    await loop.run_in_executor(None, generator.generate_pdf, history_item, pdf_path)
                    progress_cb("REPORT_GENERATION", "completed", "PDF report generated successfully")
                except Exception as e:
                    logger.error(f"Failed to generate PDF in job: {e}")
                    progress_cb("REPORT_GENERATION", "completed", "PDF generation failed")

                await job_manager.finish_job(jid, result)
            except Exception as e:
                logger.error(f"Inference error in background: {traceback.format_exc()}")
                error_msg = str(e).replace(t_path, "uploaded_file")
                await job_manager.fail_job(jid, error_msg, "INFERENCE_ERROR")
                
                # Save failure to history
                try:
                    with open(HISTORY_FILE, "r") as f:
                        history = json.load(f)
                    failure_item = {
                        "id": jid,
                        "timestamp": time.time(),
                        "filename": fname,
                        "status": "failed",
                        "error_message": error_msg,
                        "case_id": case_id,
                        "asset_hash": asset_hash
                    }
                    history.insert(0, failure_item)
                    history = history[:50]
                    with open(HISTORY_FILE, "w") as f:
                        json.dump(history, f)
                except Exception as he:
                    logger.warning(f"Failed to save failure history: {he}")
                    
            finally:
                if t_path and os.path.exists(t_path):
                    try:
                        os.remove(t_path)
                    except Exception:
                        pass
        
        # Schedule the job to run in the background
        asyncio.create_task(run_analysis_job(job_id, temp_path, filename))
        
        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

@app.post("/api/analyze-demo")
async def analyze_demo(type: str = "fake_fake"):
    """Run analysis on a pre-selected local demo file."""
    if not service or not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    paths = {
        "real_real": os.path.join(PROJECT_ROOT, "FakeAVCeleb_v1.2", "FakeAVCeleb_v1.2", "RealVideo-RealAudio", "African", "men", "id00076", "00109.mp4"),
        "real_fake": os.path.join(PROJECT_ROOT, "FakeAVCeleb_v1.2", "FakeAVCeleb_v1.2", "RealVideo-FakeAudio", "African", "men", "id00076", "00109_fake.mp4"),
        "fake_real": os.path.join(PROJECT_ROOT, "FakeAVCeleb_v1.2", "FakeAVCeleb_v1.2", "FakeVideo-RealAudio", "African", "men", "id00076", "00109_1.mp4"),
        "fake_fake": os.path.join(PROJECT_ROOT, "FakeAVCeleb_v1.2", "FakeAVCeleb_v1.2", "FakeVideo-FakeAudio", "African", "men", "id00076", "00109_10_id00476_wavtolip.mp4"),
    }
    
    demo_file = paths.get(type)
        
    if not demo_file or not os.path.exists(demo_file):
        raise HTTPException(status_code=404, detail=f"Demo file for type '{type}' not found locally.")
        
    target_file = demo_file
    demo_file_name = f"{type}.mp4" if type == "real_real" else "00109_fake.mp4"
    if type == "real_real":
        demo_file_name = "00109.mp4"
    
    logger.info(f"Running demo inference on: {target_file}")
    
    job_id = "run_" + uuid.uuid4().hex
    job_manager.create_job(job_id)
    
    async def run_demo_job(jid, t_path, fname):
        loop = asyncio.get_running_loop()
        def progress_cb(stage, status, msg, progress=None):
            asyncio.run_coroutine_threadsafe(
                job_manager.update_stage_status(jid, stage, status, msg, progress),
                loop
            )
            
        try:
            # Hash the demo file
            sha256_hash = hashlib.sha256()
            with open(t_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256_hash.update(chunk)
            asset_hash = sha256_hash.hexdigest()
            file_size = os.path.getsize(t_path)
            
            context = {
                "case_id": "case_" + asset_hash[:16],
                "asset_id": "asset_" + asset_hash[:16],
                "run_id": jid,
                "asset_hash": asset_hash,
                "file_size_bytes": file_size,
                "ingestion_timestamp": time.time()
            }
            
            result = await loop.run_in_executor(None, lambda: service.analyze_video(t_path, progress_cb, context))
            
            try:
                with open(HISTORY_FILE, "r") as f:
                    history = json.load(f)
                history_item = result.copy()
                history_item["id"] = jid
                history_item["timestamp"] = time.time()
                history_item["filename"] = fname
                history.insert(0, history_item)
                history = history[:50]
                with open(HISTORY_FILE, "w") as f:
                    json.dump(history, f)
            except Exception as e:
                logger.warning(f"Failed to save history: {e}")

            # --- REPORT GENERATION ---
            progress_cb("REPORT_GENERATION", "started", "Generating Forensic PDF Report")
            try:
                out_dir = os.path.join(PROJECT_ROOT, "experiments", "final_demo", "reports")
                os.makedirs(out_dir, exist_ok=True)
                pdf_path = os.path.join(out_dir, f"MediaDNA_Forensic_Report_{jid}.pdf")
                
                generator = ReportGenerator()
                await loop.run_in_executor(None, generator.generate_pdf, history_item, pdf_path)
                progress_cb("REPORT_GENERATION", "completed", "PDF report generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate PDF in demo job: {e}")
                progress_cb("REPORT_GENERATION", "completed", "PDF generation failed")

            await job_manager.finish_job(jid, result)
        except Exception as e:
            logger.error(f"Demo inference error: {traceback.format_exc()}")
            await job_manager.fail_job(jid, str(e), "INFERENCE_ERROR")
            
    asyncio.create_task(run_demo_job(job_id, target_file, demo_file_name))
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        # First yield the current state immediately
        yield f"data: {json.dumps(job)}\n\n"
        
        # Then listen on queue
        q = job_manager.queues.get(job_id)
        if not q:
            return
            
        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {event}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/report/{case_id}")
async def generate_report_post(case_id: str):
    """Generate a PDF forensic report for a specific case ID."""
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            
        case_data = next((h for h in history if h.get("id") == case_id), None)
        if not case_data:
            raise HTTPException(status_code=404, detail="Case not found in history.")
            
        out_dir = os.path.join(PROJECT_ROOT, "experiments", "final_demo", "reports")
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, f"MediaDNA_Forensic_Report_{case_id}.pdf")
        
        generator = ReportGenerator()
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, generator.generate_pdf, case_data, pdf_path)
        
        return {"status": "generated", "case_id": case_id, "url": f"/api/report/{case_id}"}
    except Exception as e:
        logger.error(f"Report generation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/{case_id}")
async def get_report(case_id: str, download: bool = False):
    """Download or preview an existing PDF forensic report."""
    pdf_path = os.path.join(PROJECT_ROOT, "experiments", "final_demo", "reports", f"MediaDNA_Forensic_Report_{case_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not generated yet.")
        
    return FileResponse(
        path=pdf_path,
        filename=f"MediaDNA_Forensic_Report_{case_id}.pdf",
        media_type="application/pdf",
        content_disposition_type="attachment" if download else "inline"
    )

@app.get("/api/history")
async def get_history():
    """Return local analysis history."""
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

@app.delete("/api/history")
async def clear_history():
    """Clear local analysis history."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}

@app.delete("/api/history/{item_id}")
async def delete_history_item(item_id: str):
    """Delete a specific history item."""
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        history = [h for h in history if h.get("id") != item_id]
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}

