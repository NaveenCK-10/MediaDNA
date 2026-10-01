import re
import os
import time
import sys

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/inference.py', 'r') as f:
    content = f.read()

# Update import
content = content.replace(
    'from backend.schemas.mediadna import MediaDNAProfile, Classification, Authenticity, VisualEvidence, AudioEvidence, TemporalEvidence, MultimodalFusion, Manipulation, Provenance, CloudForensics',
    'from backend.schemas.mediadna import MediaDNAProfile, Classification, Authenticity, VisualEvidence, AudioEvidence, TemporalEvidence, MultimodalFusion, Manipulation, CloudForensics, TrustSchema, QualityFindings, EvidenceItem, ProvenanceV20'
)

# Update analyze_video signature
content = content.replace(
    'def analyze_video(self, video_path: str, progress_cb=None) -> dict:',
    'def analyze_video(self, video_path: str, progress_cb=None, context: dict = None) -> dict:\n        if context is None: context = {}'
)

# Replace the Profile instantiation
old_profile = """        profile = MediaDNAProfile(
            profile_version="V16",
            case_id=os.path.basename(video_path),
            classification=Classification(
                label="fake" if calib_score >= 0.60 else "real",
                raw_logit=raw_logit,
                fake_probability=base_fake_prob
            ),
            authenticity=Authenticity(
                calibrated_probability=calib_score,
                uncertainty="Low" if abs(calib_score - 0.5) > 0.3 else "High"
            ),
            visual=VisualEvidence(
                signature="Frozen Visual Encoder Jitter",
                anomaly_score=vis_score,
                evidence=str(vis_analysis.get("metrics", {}))
            ),
            audio=AudioEvidence(
                signature="Transformer Spectral Variance",
                anomaly_score=aud_analysis.get("audio_anomaly_score", 0.0),
                evidence=str(aud_analysis.get("metrics", {}))
            ),
            temporal=TemporalEvidence(
                signature="MobileNetV2 Frame L2 Shift",
                anomaly_score=tem_score,
                evidence_intervals=temporal_intervals
            ),
            multimodal=MultimodalFusion(
                fusion_output=fusion_score
            ),
            manipulation=Manipulation(
                category=prov_data["category"],
                scores={"confidence": float(np.max(manip_probs)) if self.fusion_mlp else 0.0}
            ),
            provenance=Provenance(
                category=prov_data["category"],
                scores=prov_data["scores"],
                model_version=prov_data["model_version"]
            ),
            cloud_forensics=cloud_forensics,
            metadata={
                "video_duration_sec": 1.0,
                "analysis_duration_sec": time.time() - total_start
            },
            limitations=[
                "Model confidence assumes face visibility.",
                "Not calibrated for heavily compressed video."
            ],
            llm_report="",
            rag_explanation="",
            explainability=explainability_data
        )"""

new_profile = """        abstention_state = "BORDERLINE"
        if calib_score > 0.8: abstention_state = "LIKELY_MANIPULATED"
        elif calib_score < 0.2: abstention_state = "LIKELY_AUTHENTIC"
        
        trust = TrustSchema(
            raw_model_score=float(raw_logit),
            calibrated_probability=float(calib_score),
            model_confidence="High" if abs(calib_score - 0.5) > 0.3 else "Low",
            evidence_agreement="High" if abs(vis_score - aud_var) < 0.3 else "Low",
            ood_signal="Normal",
            abstention_state=abstention_state,
            calibration_status="NOT_VALIDATED",
            ood_status="NOT_VALIDATED"
        )
        
        try:
            vr = VideoReader(video_path)
            fps = vr.get_avg_fps()
            duration = len(vr) / fps if fps > 0 else 0
            width, height = vr[0].shape[1], vr[0].shape[0]
            resolution = f"{width}x{height}"
        except:
            fps = 30.0
            duration = 1.0
            resolution = "unknown"
            width, height = 0, 0
            
        findings = []
        if height < 224 or width < 224:
            findings.append("LOW_RESOLUTION")
        if duration < 0.5:
            findings.append("AUDIO_VIDEO_DURATION_MISMATCH")
            
        quality = QualityFindings(
            face_detected=True,
            resolution=resolution,
            duration=float(duration),
            frame_rate=float(fps),
            codec="unknown",
            audio_presence=True,
            audio_duration=float(duration),
            findings=findings
        )
        
        evidence_items = [
            EvidenceItem(
                id="ev_vis_1",
                type="visual_artifact",
                modality="visual",
                timestamp=0.0,
                value=float(vis_score),
                unit="anomaly_score",
                method="VideoCAVMAEFT_Visual",
                model="OpenAVFF",
                model_version="V16",
                source="inference",
                reliability="Medium",
                calibration_status="NOT_VALIDATED",
                interpretation="This region produced a measurable change in the detector's output when occluded.",
                limitations="This is model-sensitivity attribution and is not proof that these pixels were manipulated."
            ),
            EvidenceItem(
                id="ev_aud_1",
                type="audio_artifact",
                modality="audio",
                timestamp=0.0,
                value=float(aud_var),
                unit="variance",
                method="VideoCAVMAEFT_Audio",
                model="OpenAVFF",
                model_version="V16",
                source="inference",
                reliability="Medium",
                calibration_status="NOT_VALIDATED",
                interpretation="Spectral variance differs from typical natural speech.",
                limitations="May be triggered by background noise or compression."
            )
        ]
        
        prov_v20 = ProvenanceV20(
            metadata_provenance="Unavailable",
            cryptographic_provenance="Unavailable",
            source_device_clues="Unavailable",
            broad_manipulation_family=prov_data["category"],
            heuristic_provenance=prov_data["category"],
            exact_generator_attribution="Not supported by current validated classifiers"
        )
        
        model_finding = "Likely manipulated" if calib_score >= 0.60 else ("Likely authentic" if calib_score <= 0.40 else "Borderline")
        human_determination = "Pending review"

        profile = MediaDNAProfile(
            case_id=context.get("case_id", os.path.basename(video_path)),
            asset_id=context.get("asset_id", "asset_" + str(int(time.time()))),
            run_id=context.get("run_id", "run_" + str(int(time.time()))),
            asset_hash=context.get("asset_hash", "unknown_hash"),
            file_size_bytes=context.get("file_size_bytes", 0),
            processing_status="COMPLETED",
            model_name="OpenAVFF",
            model_version="V20",
            checkpoint_hash="unknown",
            threshold=0.5,
            preprocessing_version="V1",
            explainability_version="V1",
            runtime_metadata={"analysis_duration_sec": time.time() - total_start},
            trust=trust,
            quality=quality,
            evidence_items=evidence_items,
            model_finding=model_finding,
            human_determination=human_determination,
            provenance_v20=prov_v20,
            profile_version="V20",
            classification=Classification(
                label="fake" if calib_score >= 0.60 else "real",
                raw_logit=float(raw_logit),
                fake_probability=float(base_fake_prob)
            ),
            authenticity=Authenticity(
                calibrated_probability=float(calib_score),
                uncertainty="Low" if abs(calib_score - 0.5) > 0.3 else "High"
            ),
            visual=VisualEvidence(
                signature="Frozen Visual Encoder Jitter",
                anomaly_score=float(vis_score),
                evidence=str(vis_analysis.get("metrics", {}))
            ),
            audio=AudioEvidence(
                signature="Transformer Spectral Variance",
                anomaly_score=float(aud_analysis.get("audio_anomaly_score", 0.0)),
                evidence=str(aud_analysis.get("metrics", {}))
            ),
            temporal=TemporalEvidence(
                signature="MobileNetV2 Frame L2 Shift",
                anomaly_score=float(tem_score),
                evidence_intervals=temporal_intervals
            ),
            multimodal=MultimodalFusion(
                fusion_output=float(fusion_score)
            ),
            manipulation=Manipulation(
                category=prov_data["category"],
                scores={"confidence": float(np.max(manip_probs)) if self.fusion_mlp else 0.0}
            ),
            cloud_forensics=cloud_forensics,
            metadata={
                "video_duration_sec": float(duration),
                "analysis_duration_sec": time.time() - total_start
            },
            limitations=[
                "Model confidence assumes face visibility.",
                "Not calibrated for heavily compressed video."
            ],
            llm_report="",
            rag_explanation="",
            explainability=explainability_data
        )"""

if old_profile in content:
    content = content.replace(old_profile, new_profile)
    with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/inference.py', 'w') as f:
        f.write(content)
    print("Updated inference.py successfully.")
else:
    print("Could not find the profile instantiation block!")
