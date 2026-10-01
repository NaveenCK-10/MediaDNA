import os
import re

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/inference.py', 'r') as f:
    content = f.read()

# Add ffprobe helper method
ffprobe_helper = """    def _inspect_media(self, video_path: str):
        import json
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_format", "-show_streams",
            "-of", "json", video_path
        ]
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
                "container": container,
                "video_codec": v_codec,
                "audio_codec": a_codec,
                "width": width,
                "height": height,
                "fps": fps,
                "video_duration": v_duration,
                "audio_duration": a_duration,
                "audio_stream_count": len(a_streams),
                "audio_presence": len(a_streams) > 0
            }
        except Exception as e:
            logger.error(f"ffprobe failed: {e}")
            return None

    def analyze_video(self, video_path: str, progress_cb=None, context: dict = None) -> dict:"""

content = content.replace('    def analyze_video(self, video_path: str, progress_cb=None, context: dict = None) -> dict:', ffprobe_helper)


# Rewrite the Profile instantiation block
old_block_start = """        abstention_state = "BORDERLINE"
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
        )"""

new_block_start = """        # -------------------------------------
        # V20.1 SCIENTIFIC INTEGRITY CORRECTION
        # -------------------------------------
        media_info = self._inspect_media(video_path)
        
        findings = []
        resolution = "unknown"
        width = None
        height = None
        fps = 0.0
        v_duration = 0.0
        a_duration = 0.0
        a_presence = False
        a_stream_count = 0
        container = None
        v_codec = None
        a_codec = None
        metadata_status = "AVAILABLE"
        
        if media_info:
            width = media_info["width"]
            height = media_info["height"]
            resolution = f"{width}x{height}" if width and height else "unknown"
            fps = media_info["fps"]
            v_duration = media_info["video_duration"]
            a_duration = media_info["audio_duration"]
            a_presence = media_info["audio_presence"]
            a_stream_count = media_info["audio_stream_count"]
            container = media_info["container"]
            v_codec = media_info["video_codec"]
            a_codec = media_info["audio_codec"]
            
            if height and width and (height < 224 or width < 224):
                findings.append("LOW_RESOLUTION")
            if a_presence and abs(v_duration - a_duration) > 0.5:
                findings.append("AUDIO_VIDEO_DURATION_MISMATCH")
            if v_duration > 300:
                findings.append("EXCESSIVE_DURATION")
        else:
            metadata_status = "UNAVAILABLE"
            findings.append("UNSUPPORTED_INPUT")
        
        # Face detection fallback
        face_detected = None
        face_count = None
        face_method = None
        face_status = "NOT_IMPLEMENTED"

        # Trust & Abstention
        abstention_state = "BORDERLINE"
        
        trust = TrustSchema(
            raw_model_score=float(raw_logit),
            calibrated_probability=None,
            model_confidence=None,
            evidence_agreement=None,
            evidence_agreement_status="NOT_IMPLEMENTED",
            ood_signal=None,
            abstention_state=abstention_state,
            abstention_status="HEURISTIC",
            calibration_status="NOT_VALIDATED",
            ood_status="NOT_IMPLEMENTED"
        )
        
        quality = QualityFindings(
            face_detected=face_detected,
            face_count=face_count,
            face_detection_method=face_method,
            face_detection_status=face_status,
            resolution=resolution,
            width=width,
            height=height,
            duration=float(v_duration),
            frame_rate=float(fps),
            container=container,
            video_codec=v_codec,
            audio_codec=a_codec,
            metadata_status=metadata_status,
            audio_presence=a_presence,
            audio_duration=float(a_duration),
            audio_stream_count=a_stream_count,
            audio_status="AVAILABLE" if media_info else "UNKNOWN",
            findings=findings
        )
"""
content = content.replace(old_block_start, new_block_start)


# Update the rest of the profile
old_profile_rest = """        model_finding = "Likely manipulated" if calib_score >= 0.60 else ("Likely authentic" if calib_score <= 0.40 else "Borderline")
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
            ),"""

new_profile_rest = """        # Global Decision Threshold
        DECISION_THRESHOLD = 0.60
        model_finding = "Likely manipulated" if base_fake_prob >= DECISION_THRESHOLD else "Likely authentic"
        human_determination = "Pending review"
        
        proc_status = "COMPLETED"
        if not context.get("asset_hash"):
            proc_status = "INTEGRITY_CHECK_FAILED"

        profile = MediaDNAProfile(
            case_id=context.get("case_id", "UNKNOWN"),
            asset_id=context.get("asset_id", "UNKNOWN"),
            run_id=context.get("run_id", "UNKNOWN"),
            asset_hash=context.get("asset_hash", "UNAVAILABLE"),
            file_size_bytes=context.get("file_size_bytes", -1),
            ingestion_timestamp=context.get("ingestion_timestamp", time.time()),
            processing_status=proc_status,
            model_name="OpenAVFF",
            model_version="V20.1",
            checkpoint_hash=None,
            checkpoint_hash_status="UNAVAILABLE",
            decision_protocol_version="V20.1",
            threshold=DECISION_THRESHOLD,
            preprocessing_version="V1",
            explainability_version="V1",
            runtime_metadata={"analysis_duration_sec": time.time() - total_start},
            trust=trust,
            quality=quality,
            evidence_items=evidence_items,
            model_finding=model_finding,
            human_determination=human_determination,
            provenance_v20=prov_v20,
            profile_version="V20.1",
            classification=Classification(
                label="fake" if base_fake_prob >= DECISION_THRESHOLD else "real",
                raw_logit=float(raw_logit),
                fake_probability=float(base_fake_prob),
                decision_threshold=DECISION_THRESHOLD
            ),
            authenticity=Authenticity(
                calibrated_probability=None,
                uncertainty="UNKNOWN"
            ),"""

content = content.replace(old_profile_rest, new_profile_rest)

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/backend/inference.py', 'w') as f:
    f.write(content)

print("Updated inference.py successfully.")
