export interface TrustSchema {
  raw_model_score: number;
  calibrated_score?: number | null;
  model_confidence?: string | null;
  evidence_agreement?: string | null;
  evidence_agreement_status: string;
  ood_signal?: string | null;
  abstention_state: string;
  abstention_status: string;
  calibration_status: string;
  ood_status: string;
}

export interface QualityFindings {
  face_detected?: boolean | null;
  face_count?: number | null;
  face_detection_method?: string | null;
  face_detection_status: string;
  
  resolution: string;
  width?: number | null;
  height?: number | null;
  duration: number;
  frame_rate: number;
  
  container?: string | null;
  video_codec?: string | null;
  audio_codec?: string | null;
  metadata_status: string;
  
  audio_presence?: boolean | null;
  audio_duration?: number | null;
  audio_stream_count?: number | null;
  audio_status: string;
  
  findings: string[];
}

export interface EvidenceItem {
  id: string;
  type: string;
  modality: string;
  timestamp: number;
  frame?: number | null;
  region?: string | null;
  value: number;
  unit: string;
  method: string;
  model: string;
  model_version: string;
  source: string;
  reliability: string;
  calibration_status: string;
  interpretation: string;
  limitations: string;
}

export interface ProvenanceV20 {
  metadata_provenance: string;
  cryptographic_provenance: string;
  source_device_clues: string;
  broad_manipulation_family: string;
  heuristic_provenance: string;
  exact_generator_attribution: string;
}

export interface Classification {
  label: string;
  raw_logit: number;
  decision_score: number;
  decision_threshold: number;
}

export interface Authenticity {
  calibrated_score?: number | null;
  uncertainty: string;
}

export interface VisualEvidence {
  signature: string;
  anomaly_score: number;
  evidence: string;
}

export interface AudioEvidence {
  signature: string;
  anomaly_score: number;
  evidence: string;
}

export interface TemporalEvidence {
  signature: string;
  anomaly_score: number;
  evidence_intervals: string[];
}

export interface MultimodalFusion {
  fusion_output: number;
}

export interface Manipulation {
  category: string;
  scores: Record<string, number>;
}

export interface CloudForensics {
  synthetic_video_score?: number | null;
  active_speaker_count?: number | null;
  whisper_transcription?: string | null;
}

export interface ExplainabilityData {
  narrative: {
    mode: string;
    text: string;
  };
  modality_sensitivity: {
    visual_only: number;
    audio_only: number;
  };
  av_alignment: {
    correlation_score: number;
    interpretation: string;
  };
  region_sensitivity?: Array<{
    rank: number;
    region: string;
    original_score: number;
    perturbed_score: number;
    delta: number;
  }>;
}

export interface MediaDNAProfile {
  case_id: string;
  asset_id: string;
  run_id: string;
  asset_hash: string;
  file_size_bytes: number;
  ingestion_timestamp: number;
  processing_status: string;
  
  model_name: string;
  model_version: string;
  checkpoint_hash?: string | null;
  checkpoint_hash_status: string;
  decision_protocol_version: string;
  
  threshold: number;
  preprocessing_version: string;
  explainability_version: string;
  runtime_metadata: Record<string, any>;
  
  trust: TrustSchema;
  quality: QualityFindings;
  evidence_items: EvidenceItem[];
  
  model_finding: string;
  human_determination: string;
  provenance_v20: ProvenanceV20;
  
  profile_version: string;
  classification: Classification;
  authenticity: Authenticity;
  visual?: VisualEvidence | null;
  audio?: AudioEvidence | null;
  temporal?: TemporalEvidence | null;
  multimodal?: MultimodalFusion | null;
  manipulation?: Manipulation | null;
  cloud_forensics?: CloudForensics | null;
  metadata: Record<string, any>;
  limitations: string[];
  llm_report: string;
  rag_explanation: string;
  explainability?: ExplainabilityData | null;
}

export type AnalysisResponse = MediaDNAProfile;

export interface HistoryItem extends MediaDNAProfile {
  id: string;
  timestamp: number;
  filename?: string;
  video_filename?: string;
}

export interface HealthData {
  status: string;
  model_loaded: boolean;
  device: string;
  checkpoint: string;
  gpu_name: string | null;
  gpu_memory_mb: number | null;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  n_classes: number;
  input_visual: string;
  input_audio: string;
  num_frames: number;
  audio_sample_rate: number;
  target_length: number;
  num_mel_bins: number;
  dataset_mean: number;
  dataset_std: number;
  checkpoint: string;
  device: string;
  total_parameters: number;
}

export type HealthResponse = HealthData;
export type ModelInfoResponse = ModelInfo;
export type HistoryEntry = HistoryItem;

export interface JobSubmissionResponse {
  job_id: string;
  status: string;
}

export interface JobEvent {
  job_id: string;
  status: 'queued' | 'processing' | 'complete' | 'error';
  stage: string;
  message: string;
  progress: number;
  elapsed_seconds: number;
  completed_stages: string[];
  result?: MediaDNAProfile;
  error_code?: string;
}
