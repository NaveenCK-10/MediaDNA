from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class TrustSchema(BaseModel):
    raw_model_score: float
    calibrated_probability: Optional[float] = None
    model_confidence: Optional[str] = None
    evidence_agreement: Optional[str] = None
    evidence_agreement_status: str = "NOT_IMPLEMENTED"
    ood_signal: Optional[str] = None
    abstention_state: str
    abstention_status: str
    calibration_status: str = "NOT_VALIDATED"
    ood_status: str = "NOT_IMPLEMENTED"

class QualityFindings(BaseModel):
    face_detected: Optional[bool] = None
    face_count: Optional[int] = None
    face_detection_method: Optional[str] = None
    face_detection_status: str = "NOT_IMPLEMENTED"
    
    resolution: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration: float
    frame_rate: float
    
    container: Optional[str] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    metadata_status: str = "AVAILABLE"
    
    audio_presence: Optional[bool] = None
    audio_duration: Optional[float] = None
    audio_stream_count: Optional[int] = None
    audio_status: str = "UNKNOWN"
    
    findings: List[str]

class EvidenceItem(BaseModel):
    id: str
    type: str
    modality: str
    timestamp: float
    frame: Optional[int] = None
    region: Optional[str] = None
    value: float
    unit: str
    method: str
    model: str
    model_version: str
    source: str
    reliability: str
    calibration_status: str
    interpretation: str
    limitations: str

class ProvenanceV20(BaseModel):
    metadata_provenance: str
    cryptographic_provenance: str
    source_device_clues: str
    broad_manipulation_family: str
    heuristic_provenance: str

# Legacy models kept for backwards compatibility during transition, or re-mapped.
class Classification(BaseModel):
    label: str
    raw_logit: float
    fake_probability: float
    decision_threshold: float = 0.60

class Authenticity(BaseModel):
    calibrated_probability: Optional[float] = None
    uncertainty: str

class VisualEvidence(BaseModel):
    signature: str
    anomaly_score: float
    evidence: str

class AudioEvidence(BaseModel):
    signature: str
    anomaly_score: float
    evidence: str

class TemporalEvidence(BaseModel):
    signature: str
    anomaly_score: float
    evidence_intervals: List[str]

class MultimodalFusion(BaseModel):
    fusion_output: float

class Manipulation(BaseModel):
    category: str
    scores: Dict[str, float]

class CloudForensics(BaseModel):
    synthetic_video_probability: Optional[float] = None
    active_speaker_count: Optional[int] = None
    whisper_transcription: Optional[str] = None

class ExplainabilityData(BaseModel):
    key_frames: List[int] = []
    visual_attributions: List[Dict[str, Any]] = []
    region_sensitivity: List[Dict[str, Any]] = []
    temporal_events: List[Dict[str, Any]] = []
    audio_attributions: List[Dict[str, Any]] = []
    av_alignment: Dict[str, Any] = {}
    modality_sensitivity: Dict[str, float] = {}
    evidence_contributions: List[Dict[str, Any]] = []
    fingerprint: Dict[str, Any] = {}
    narrative: Dict[str, str] = {}

class MediaDNAProfile(BaseModel):
    case_id: str
    asset_id: str
    run_id: str
    asset_hash: str # SHA-256
    file_size_bytes: int
    ingestion_timestamp: float
    processing_status: str
    
    model_name: str
    model_version: str
    checkpoint_hash: Optional[str] = None
    checkpoint_hash_status: str = "UNAVAILABLE"
    decision_protocol_version: str = "V20.1"
    
    threshold: float
    preprocessing_version: str
    explainability_version: str
    runtime_metadata: Dict[str, Any]
    
    trust: TrustSchema
    quality: QualityFindings
    evidence_items: List[EvidenceItem]
    
    model_finding: str
    human_determination: str
    provenance_v20: ProvenanceV20
    
    profile_version: str
    classification: Classification
    authenticity: Authenticity
    visual: Optional[VisualEvidence] = None
    audio: Optional[AudioEvidence] = None
    temporal: Optional[TemporalEvidence] = None
    multimodal: Optional[MultimodalFusion] = None
    manipulation: Optional[Manipulation] = None
    cloud_forensics: Optional[CloudForensics] = None
    metadata: dict
    limitations: List[str]
    llm_report: str
    rag_explanation: str
    explainability: Optional[ExplainabilityData] = None
