export interface VisualSignals {
  blur_variance_mean: number;
  blur_coefficient_of_variation: number;
  frame_mae_mean: number;
  frame_mae_std: number;
  visual_anomaly_score: number;
  frames_analyzed: number;
}

export interface Metadata {
  duration?: string;
  resolution?: string;
  fps?: string;
  video_codec?: string;
  audio_codec?: string;
  bitrate?: string;
  file_size?: string;
  source?: string;
}

export interface AnalysisResponse {
  prediction: "fake" | "real";
  fake_probability: number;
  real_probability: number;
  openavff_fake_prob: number;
  openavff_real_prob: number;
  raw_logits: number[];
  visual_signals: VisualSignals;
  metadata: Metadata;
  model: string;
  checkpoint: string;
  device: string;
  inference_time: number;
  total_time: number;
  frames_processed: number;
  audio_sample_rate: number;
  video_filename: string;
}

export interface HistoryItem extends AnalysisResponse {
  id: string;
  timestamp: number;
  filename?: string;
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
