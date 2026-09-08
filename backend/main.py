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
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import OpenAVFFService, DATASET_MEAN, DATASET_STD, TARGET_LENGTH, NUM_MEL_BINS, NUM_FRAMES, AUDIO_SAMPLE_RATE, IM_RES
from backend.schemas import AnalysisResponse, HealthResponse, ModelInfoResponse, ErrorResponse
from backend.modules import analyze_visual_signals, extract_metadata, calculate_fusion
import json

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
        input_visual=f"{IM_RES}×{IM_RES}",
        input_audio=f"{TARGET_LENGTH}×{NUM_MEL_BINS} mel filterbank",
        num_frames=NUM_FRAMES,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        target_length=TARGET_LENGTH,
        num_mel_bins=NUM_MEL_BINS,
        dataset_mean=DATASET_MEAN,
        dataset_std=DATASET_STD,
        checkpoint=os.path.basename(CHECKPOINT),
        device=str(service.device),
        total_parameters=service.total_params,
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
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
        with open(temp_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB",
                    )
                f.write(chunk)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        logger.info(f"Received video: {filename} ({file_size / (1024*1024):.1f} MB)")

        # ── Run Inference & Analysis ──
        try:
            # 1. OpenAVFF Inference
            result = service.analyze_video(temp_path)
            
            # 2. Visual Analysis
            visual_signals = analyze_visual_signals(temp_path)
            
            # 3. Metadata Extraction
            metadata = extract_metadata(temp_path)
            
            # 4. MediaDNA Fusion
            fusion = calculate_fusion(result.fake_probability, visual_signals["visual_anomaly_score"])
            
        except RuntimeError as e:
            error_msg = str(e).replace(temp_path, "uploaded_file") if temp_path else str(e)
            if "FFmpeg" in error_msg:
                raise HTTPException(status_code=422, detail=f"Audio extraction failed: {error_msg}")
            elif "frame" in error_msg.lower() or "video" in error_msg.lower():
                raise HTTPException(status_code=422, detail=f"Video processing failed: {error_msg}")
            else:
                logger.error(f"Inference error: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Model inference failed: {error_msg}")
        except FileNotFoundError as e:
            error_msg = str(e).replace(temp_path, "uploaded_file") if temp_path else str(e)
            raise HTTPException(status_code=404, detail=error_msg)
        except Exception as e:
            logger.error(f"Unexpected inference error: {traceback.format_exc()}")
            error_msg = str(e).replace(temp_path, "uploaded_file") if temp_path else str(e)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")

        # ── Build response ──
        response_data = AnalysisResponse(
            prediction=fusion["prediction"],
            fake_probability=fusion["mediadna_fake_prob"],
            real_probability=fusion["mediadna_real_prob"],
            openavff_fake_prob=result.fake_probability,
            openavff_real_prob=result.real_probability,
            raw_logits=result.raw_logits,
            visual_signals=visual_signals,
            metadata=metadata,
            model="MediaDNA / OpenAVFF",
            checkpoint=os.path.basename(CHECKPOINT),
            device=str(service.device),
            inference_time=result.inference_time,
            total_time=result.total_time,
            frames_processed=result.frames_processed,
            audio_sample_rate=result.audio_sample_rate,
            video_filename=filename,
        )
        
        # ── Save History ──
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            
            history_item = response_data.model_dump()
            history_item["id"] = str(time.time())
            history_item["timestamp"] = time.time()
            history_item["filename"] = filename # Add filename for backwards compatibility with the history page
            
            history.insert(0, history_item)
            
            # Keep only last 50 items to avoid infinite growth
            history = history[:50]
            
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
            
        return response_data

    finally:
        # ── Cleanup temp file ──
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception:
                logger.warning(f"Failed to clean up temp file: {temp_path}")

@app.post("/api/analyze-demo", response_model=AnalysisResponse)
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
        
    filename = os.path.basename(demo_file)
    logger.info(f"Running demo inference on: {demo_file}")
    
    try:
        # 1. OpenAVFF Inference
        result = service.analyze_video(demo_file)
        # 2. Visual Analysis
        visual_signals = analyze_visual_signals(demo_file)
        # 3. Metadata Extraction
        metadata = extract_metadata(demo_file)
        # 4. MediaDNA Fusion
        fusion = calculate_fusion(result.fake_probability, visual_signals["visual_anomaly_score"])
    except Exception as e:
        logger.error(f"Demo inference error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    response_data = AnalysisResponse(
        prediction=fusion["prediction"],
        fake_probability=fusion["mediadna_fake_prob"],
        real_probability=fusion["mediadna_real_prob"],
        openavff_fake_prob=result.fake_probability,
        openavff_real_prob=result.real_probability,
        raw_logits=result.raw_logits,
        visual_signals=visual_signals,
        metadata=metadata,
        model="MediaDNA / OpenAVFF",
        checkpoint=os.path.basename(CHECKPOINT),
        device=str(service.device),
        inference_time=result.inference_time,
        total_time=result.total_time,
        frames_processed=result.frames_processed,
        audio_sample_rate=result.audio_sample_rate,
        video_filename=f"[DEMO] {filename}",
    )
    
    # Save History
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            
        history_item = response_data.model_dump()
        history_item["id"] = str(time.time())
        history_item["timestamp"] = time.time()
        history_item["filename"] = f"[DEMO] {filename}"
        
        history.insert(0, history_item)
        
        # Keep only last 50 items to avoid infinite growth
        history = history[:50]
        
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")
        
    return response_data

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

