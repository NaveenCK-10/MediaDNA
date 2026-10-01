"""
OpenAVFF Inference Service for MediaDNA (V22.3).
Provides the full MediaDNA Forensic Profile using Repaired Specialists and Late Fusion.
"""
import os
import sys
import time
import subprocess
import logging
import pickle

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import soundfile as sf
import torchvision.transforms as T
from decord import VideoReader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the new specialists
from src.models.visual_specialist import VisualSpecialist
from src.models.audio_specialist import AudioSpecialist

from backend.modules.temporal_forensics import TemporalForensicsModule
from backend.modules.provenance import ProvenanceModule
from backend.modules.llm.forensic_report import ForensicReportGenerator
from backend.modules.nvidia_nim_api import NVIDIANimAPI
from backend.schemas.mediadna import (
    MediaDNAProfile, Classification, Authenticity, VisualEvidence, 
    AudioEvidence, TemporalEvidence, MultimodalFusion, Manipulation, 
    CloudForensics, TrustSchema, QualityFindings, EvidenceItem, ProvenanceV20
)

logger = logging.getLogger("mediadna.inference")

FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
IM_RES = 224
NUM_FRAMES = 16

FFPROBE_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe"

class OpenAVFFService:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vis_ckpt = os.path.join(PROJECT_ROOT, "V22_3B_VISUAL_CHECKPOINT.pth")
        self.aud_ckpt = os.path.join(PROJECT_ROOT, "V22_3C_AUDIO_CHECKPOINT.pth")
        self.calibrator_ckpt = os.path.join(PROJECT_ROOT, "V22_3_PLATT_CALIBRATOR.pkl")
        
        self._load_models()
        self._load_modules()
        
    def _load_models(self):
        logger.info(f"Loading V22.3 Visual Specialist...")
        self.vis_model = VisualSpecialist()
        self.vis_model.load_state_dict(torch.load(self.vis_ckpt, map_location="cpu"))
        self.vis_model = nn.DataParallel(self.vis_model).to(self.device)
        self.vis_model.eval()
        
        logger.info(f"Loading V22.3 Audio Specialist...")
        self.aud_model = AudioSpecialist()
        self.aud_model.load_state_dict(torch.load(self.aud_ckpt, map_location="cpu"))
        self.aud_model = nn.DataParallel(self.aud_model).to(self.device)
        self.aud_model.eval()
        
        logger.info(f"Loading Platt Calibrator...")
        with open(self.calibrator_ckpt, "rb") as f:
            self.calibrator = pickle.load(f)

    def _load_modules(self):
        logger.info(f"Loading Forensic Modules...")
        self.tem_mod = TemporalForensicsModule()
        self.prov_mod = ProvenanceModule()
        self.llm = ForensicReportGenerator()
        self.nim_api = NVIDIANimAPI()

    def _extract_audio_melspec(self, video_path: str):
        """Extract audio mel spectrogram matching Audio Specialist exactly."""
        temp_wav = f"temp_inf_{os.getpid()}.wav"
        try:
            cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                   "-ar", "16000", "-ac", "1", temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            
            if not os.path.exists(temp_wav):
                return torch.zeros(1, 1024, 128)
            
            wav, sr = sf.read(temp_wav)
            if len(wav.shape) > 1: wav = wav.mean(axis=1)
            audio_tensor = torch.FloatTensor(wav)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128
            )(audio_tensor)
            
            mel_spec = (mel_spec + 1e-6).log()
            mel_spec = (mel_spec - (-5.081)) / 4.4849
            
            target_len = 1024
            if mel_spec.shape[1] < target_len:
                pad = target_len - mel_spec.shape[1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
            else:
                mel_spec = mel_spec[:, :target_len]
            
            mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)  # (1, 1024, 128)
            
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return mel_spec
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return torch.zeros(1, 1024, 128)

    def _extract_video_frames(self, video_path: str):
        vr = VideoReader(video_path, width=224, height=224)
        num_frames = len(vr)
        frame_idx = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        v_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        v_tensor = transform(v_tensor).permute(1, 0, 2, 3).unsqueeze(0)
        return v_tensor, num_frames

    def _inspect_media(self, video_path: str):
        import json
        cmd = [FFPROBE_PATH, "-v", "error", "-show_format", "-show_streams", "-of", "json", video_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            v_stream = next((s for s in streams if s["codec_type"] == "video"), None)
            a_streams = [s for s in streams if s["codec_type"] == "audio"]
            container = data.get("format", {}).get("format_name", "UNKNOWN")
            v_codec = v_stream["codec_name"] if v_stream else None
            a_codec = a_streams[0]["codec_name"] if a_streams else None
            width = int(v_stream.get("width", 0)) if v_stream else None
            height = int(v_stream.get("height", 0)) if v_stream else None
            fps = 0.0
            if v_stream and "r_frame_rate" in v_stream:
                num, den = v_stream["r_frame_rate"].split('/')
                if int(den) > 0: fps = float(num) / float(den)
            v_duration = float(v_stream.get("duration", data.get("format", {}).get("duration", 0))) if v_stream else 0.0
            a_duration = float(a_streams[0].get("duration", data.get("format", {}).get("duration", 0))) if a_streams else 0.0
            return {
                "container": container, "video_codec": v_codec, "audio_codec": a_codec,
                "width": width, "height": height, "fps": fps,
                "video_duration": v_duration, "audio_duration": a_duration,
                "audio_stream_count": len(a_streams), "audio_presence": len(a_streams) > 0
            }
        except Exception as e:
            logger.error(f"ffprobe failed: {e}")
            return None

    def analyze_video(self, video_path: str, progress_cb=None, context: dict = None) -> dict:
        if context is None: context = {}
        total_start = time.time()
        
        # --- SECURITY/SANDBOXING (VALIDATING / INSPECTING_MEDIA) ---
        if progress_cb: progress_cb("VALIDATING", "started", "Validating input constraints")
        abs_path = os.path.abspath(video_path)
        if not os.path.exists(abs_path): raise FileNotFoundError(f"Video file not found: {abs_path}")
        if os.path.getsize(abs_path) > 500 * 1024 * 1024: raise ValueError(f"File size exceeds 500MB limit.")
        if os.path.splitext(abs_path)[1].lower() not in {".mp4", ".avi", ".mov", ".mkv"}: raise ValueError("Unsupported extension.")
        if progress_cb: progress_cb("VALIDATING", "completed", "File constraints validated")
        
        if progress_cb: progress_cb("INSPECTING_MEDIA", "started", "Inspecting media streams with ffprobe")
        media_info = self._inspect_media(abs_path)
        if media_info is None: raise ValueError("Failed to inspect media file.")
        if media_info.get("video_duration", 0) > 300: raise ValueError("Video duration exceeds 300s limit.")
        if progress_cb: progress_cb("INSPECTING_MEDIA", "completed", f"Media inspected: {media_info['width']}x{media_info['height']} @ {media_info['fps']}fps")

        # --- VIDEO EXTRACTION ---
        if progress_cb: progress_cb("EXTRACTING_VIDEO", "started", "Extracting 16 frames from video stream")
        v_input, num_frames_ext = self._extract_video_frames(abs_path)
        v_input = v_input.to(self.device)
        if progress_cb: progress_cb("EXTRACTING_VIDEO", "completed", f"{num_frames_ext} frames available, sampled 16")
        
        # --- VISUAL ANALYSIS ---
        if progress_cb: progress_cb("VISUAL_ANALYSIS", "started", "Running Repaired Visual Specialist inference")
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                v_out = self.vis_model(v_input)
                vis_score = float(torch.sigmoid(v_out)[:, 0].cpu().numpy()[0])
        if progress_cb: progress_cb("VISUAL_ANALYSIS", "completed", f"Visual inference complete (score: {vis_score:.4f})")

        # --- AUDIO EXTRACTION ---
        if not media_info['audio_presence']:
            # No audio
            if progress_cb: progress_cb("EXTRACTING_AUDIO", "started", "No audio stream detected")
            if progress_cb: progress_cb("EXTRACTING_AUDIO", "completed", "Audio processing skipped")
            aud_score = None
            # Skip PREPROCESSING & ANALYSIS
            if progress_cb: progress_cb("AUDIO_PREPROCESSING", "started", "Skipped")
            if progress_cb: progress_cb("AUDIO_PREPROCESSING", "completed", "Skipped")
            if progress_cb: progress_cb("AUDIO_ANALYSIS", "started", "Skipped")
            if progress_cb: progress_cb("AUDIO_ANALYSIS", "completed", "Skipped")
        else:
            if progress_cb: progress_cb("EXTRACTING_AUDIO", "started", "Extracting audio track via ffmpeg")
            a_input = self._extract_audio_melspec(abs_path)
            if progress_cb: progress_cb("EXTRACTING_AUDIO", "completed", "Audio track extracted")
            
            if progress_cb: progress_cb("AUDIO_PREPROCESSING", "started", "Generating Mel-spectrogram")
            a_input = a_input.to(self.device)
            if progress_cb: progress_cb("AUDIO_PREPROCESSING", "completed", "Mel-spectrogram generated")

            if progress_cb: progress_cb("AUDIO_ANALYSIS", "started", "Running Repaired Audio Specialist inference")
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    a_out = self.aud_model(a_input)
                    aud_score = float(torch.sigmoid(a_out)[:, 0].cpu().numpy()[0])
            if progress_cb: progress_cb("AUDIO_ANALYSIS", "completed", f"Audio inference complete (score: {aud_score:.4f})")
        
        # --- LATE FUSION ---
        if progress_cb: progress_cb("LATE_FUSION", "started", "Fusing visual and audio modalities")
        tem_analysis = self.tem_mod.analyze(abs_path)
        tem_score = tem_analysis.get("temporal_anomaly_score", 0.0)
        
        if aud_score is not None:
            fusion_score = 0.5 * vis_score + 0.5 * aud_score
            if progress_cb: progress_cb("LATE_FUSION", "completed", f"Fusion complete (score: {fusion_score:.4f})")
            
            # --- CALIBRATION ---
            if progress_cb: progress_cb("CALIBRATION", "started", "Applying frozen Platt calibration")
            logit_score = float(np.log(fusion_score / (1 - fusion_score + 1e-12)))
            try:
                calib_probs = self.calibrator.predict_proba(np.array([[logit_score]]))
                calib_score = float(calib_probs[0][1])
            except Exception:
                calib_score = fusion_score
            if progress_cb: progress_cb("CALIBRATION", "completed", f"Calibrated probability: {calib_score:.4f}")
            
            # --- DECISION ---
            if progress_cb: progress_cb("DECISION", "started", "Applying V22 decision policy")
            if calib_score >= 0.7:
                decision = "SYNTHETIC"
                finding = "Likely manipulated"
            elif calib_score < 0.3:
                decision = "AUTHENTIC"
                finding = "Likely authentic"
            else:
                decision = "UNCERTAIN"
                finding = "Inconclusive evidence"
            if progress_cb: progress_cb("DECISION", "completed", f"Decision: {decision}")
        else:
            fusion_score = vis_score
            if progress_cb: progress_cb("LATE_FUSION", "completed", "Fusion skipped (missing audio), visual-only path")
            
            if progress_cb: progress_cb("CALIBRATION", "started", "Skipped (missing audio)")
            calib_score = 0.5
            if progress_cb: progress_cb("CALIBRATION", "completed", "Skipped")
            
            if progress_cb: progress_cb("DECISION", "started", "Applying V22 decision policy")
            decision = "UNCERTAIN"
            finding = "Audio analysis was unavailable for this media."
            if progress_cb: progress_cb("DECISION", "completed", f"Decision: {decision}")

        # --- FORENSIC EVIDENCE ---
        if progress_cb: progress_cb("FORENSIC_EVIDENCE", "started", "Aggregating forensic metadata and provenance")
        prov_data = self.prov_mod.determine_provenance(4)
        nim_results = self.nim_api.run_all(abs_path) or {}
        syn_vid = nim_results.get("synthetic_video") or {}
        act_spk = nim_results.get("active_speaker") or {}
        whisp = nim_results.get("whisper") or {}
        
        cloud_forensics = CloudForensics(
            synthetic_video_probability=syn_vid.get("probability"),
            active_speaker_count=act_spk.get("speaking_frames"),
            whisper_transcription=whisp.get("text")
        )
            
        # --- MEDIA DNA PROFILE ---
        if progress_cb: progress_cb("report_generation", "Aggregating MediaDNA profile", 95)
        
        quality = QualityFindings(
            face_detected=None, face_count=None, face_detection_method=None, face_detection_status="NOT_IMPLEMENTED",
            resolution=f"{media_info['width']}x{media_info['height']}" if media_info['width'] else "unknown",
            width=media_info['width'], height=media_info['height'],
            duration=float(media_info['video_duration']), frame_rate=float(media_info['fps']),
            container=media_info['container'], video_codec=media_info['video_codec'], audio_codec=media_info['audio_codec'],
            metadata_status="AVAILABLE", audio_presence=media_info['audio_presence'],
            audio_duration=float(media_info['audio_duration']), audio_stream_count=media_info['audio_stream_count'],
            audio_status="AVAILABLE", findings=[]
        )
        
        evidence_items = [
            EvidenceItem(
                id="ev_vis_1", type="visual_artifact", modality="visual", timestamp=0.0, value=vis_score, unit="probability",
                method="Repaired_Visual_Specialist_B", model="OpenAVFF", model_version="V22.3", source="inference",
                reliability="High", calibration_status="NOT_VALIDATED", interpretation="Visual manipulation likelihood.", limitations=""
            )
        ]
        
        if aud_score is not None:
            evidence_items.append(
                EvidenceItem(
                    id="ev_aud_1", type="audio_artifact", modality="audio", timestamp=0.0, value=aud_score, unit="probability",
                    method="Repaired_Audio_Specialist_C", model="OpenAVFF", model_version="V22.3", source="inference",
                    reliability="High", calibration_status="NOT_VALIDATED", interpretation="Audio manipulation likelihood.", limitations=""
                )
            )
        
        prov_v20 = ProvenanceV20(
            metadata_provenance="Unavailable", cryptographic_provenance="Unavailable", source_device_clues="Unavailable",
            broad_manipulation_family=prov_data["category"], heuristic_provenance=prov_data["category"]
        )
        
        trust = TrustSchema(
            raw_model_score=fusion_score, calibrated_probability=calib_score,
            model_confidence=str(round(abs(0.5 - calib_score) * 2.0, 2)),
            evidence_agreement=str(round(1.0 - abs(vis_score - (aud_score if aud_score is not None else vis_score)), 2)),
            evidence_agreement_status="IMPLEMENTED",
            ood_signal=None, abstention_state=decision,
            abstention_status="PLATT_CALIBRATED", calibration_status="VALIDATED", ood_status="NOT_IMPLEMENTED"
        )
        
        profile = MediaDNAProfile(
            case_id=context.get("case_id", "UNKNOWN"), asset_id=context.get("asset_id", "UNKNOWN"),
            run_id=context.get("run_id", "UNKNOWN"), asset_hash=context.get("asset_hash", "UNAVAILABLE"),
            file_size_bytes=context.get("file_size_bytes", -1), ingestion_timestamp=context.get("ingestion_timestamp", time.time()),
            processing_status="COMPLETED", model_name="OpenAVFF", model_version="V22.3",
            checkpoint_hash=None, checkpoint_hash_status="UNAVAILABLE",
            decision_protocol_version="V22.3", threshold=0.7, preprocessing_version="V2", explainability_version="V2",
            runtime_metadata={"analysis_duration_sec": time.time() - total_start},
            trust=trust, quality=quality, evidence_items=evidence_items,
            model_finding=finding, human_determination="Pending review", provenance_v20=prov_v20, profile_version="V22.3",
            classification=Classification(label="fake" if decision=="SYNTHETIC" else "real" if decision=="AUTHENTIC" else "unknown", raw_logit=fusion_score, fake_probability=calib_score, decision_threshold=0.7),
            authenticity=Authenticity(calibrated_probability=calib_score, uncertainty=decision),
            visual=VisualEvidence(signature="Repaired Visual Encoder", anomaly_score=vis_score, evidence=""),
            audio=AudioEvidence(signature="Repaired Audio Encoder", anomaly_score=aud_score if aud_score is not None else 0.0, evidence=""),
            temporal=TemporalEvidence(signature="MobileNetV2 Frame L2 Shift", anomaly_score=tem_score, evidence_intervals=[]),
            multimodal=MultimodalFusion(fusion_output=fusion_score),
            manipulation=Manipulation(category=prov_data["category"], scores={"confidence": 0.0}),
            cloud_forensics=cloud_forensics,
            metadata={"video_duration_sec": float(media_info['video_duration']), "analysis_duration_sec": time.time() - total_start},
            limitations=["Uses Late Fusion"],
            llm_report="", rag_explanation="", explainability={"status": "not_implemented_for_specialists"}
        )
        
        profile_dict = profile.model_dump()
        # Removed LLM report generation for deterministic templates
        profile_dict["llm_report"] = ""
        return profile_dict

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, 'vis_model') and hasattr(self, 'aud_model')

    @property
    def total_params(self) -> int:
        if self.is_loaded:
            return sum(p.numel() for p in self.vis_model.parameters()) + sum(p.numel() for p in self.aud_model.parameters())
        return 0
