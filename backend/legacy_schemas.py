"""
Pydantic response models for the MediaDNA API.
"""
from pydantic import BaseModel
from typing import Optional, List


class AnalysisResponse(BaseModel):
    """Response from the /api/analyze endpoint."""
    prediction: str  # "fake" or "real"
    fake_probability: float
    real_probability: float
    # OpenAVFF specific
    openavff_fake_prob: float
    openavff_real_prob: float
    raw_logits: List[float]
    
    # Visual specific
    visual_signals: dict
    
    # Metadata specific
    metadata: dict
    
    # Inference specifics
    model: str
    checkpoint: str
    device: str
    inference_time: float  # seconds
    total_time: float  # seconds (includes preprocessing)
    frames_processed: int
    audio_sample_rate: int
    video_filename: str


class HealthResponse(BaseModel):
    """Response from the /api/health endpoint."""
    status: str
    model_loaded: bool
    device: str
    checkpoint: str
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Response from the /api/model-info endpoint."""
    model_name: str
    architecture: str
    n_classes: int
    input_visual: str
    input_audio: str
    num_frames: int
    audio_sample_rate: int
    target_length: int
    num_mel_bins: int
    dataset_mean: float
    dataset_std: float
    checkpoint: str
    device: str
    total_parameters: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
