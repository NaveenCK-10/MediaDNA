import os
import sys
import grpc
import pathlib
import math
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NIM_CLIENTS_DIR = os.path.join(PROJECT_ROOT, "scratch", "nim-clients")

sys.path.insert(0, os.path.join(NIM_CLIENTS_DIR, "synthetic-video-detector", "interfaces"))
sys.path.insert(0, os.path.join(NIM_CLIENTS_DIR, "active-speaker-detection", "interfaces"))

import syntheticvideodetector_pb2
import syntheticvideodetector_pb2_grpc

from nvidia.ai4m.activespeakerdetection.v1 import activespeakerdetection_pb2
from nvidia.ai4m.activespeakerdetection.v1 import activespeakerdetection_pb2_grpc
from nvidia.ai4m.video.v1 import video_pb2

DATA_CHUNK_SIZE = 64 * 1024

class NVIDIANimAPI:
    def __init__(self):
        self.synthetic_key = os.environ.get("NVIDIA_SYNTHETIC_API_KEY")
        self.active_speaker_key = os.environ.get("NVIDIA_ACTIVESPEAKER_API_KEY")
        self.whisper_key = os.environ.get("NVIDIA_WHISPER_API_KEY")
        self.nemotron_key = os.environ.get("NVIDIA_NEMOTRON_API_KEY")
        self.target = "grpc.nvcf.nvidia.com:443"
        self.synthetic_function_id = "847b6e53-0133-452d-ab85-d7acf3ace723"
        # The active speaker detector has its own function ID. I'll need to figure out its function ID.
        # Actually, let me check active-speaker-detection.py to see if it lists a default.
        self.active_speaker_function_id = "0da95568-12d7-4663-8f0a-66236baf650b" # Just a placeholder if not provided

    def _generate_synthetic_request(self, video_filepath):
        with open(video_filepath, "rb") as f:
            while True:
                chunk = f.read(DATA_CHUNK_SIZE)
                if not chunk:
                    break
                yield syntheticvideodetector_pb2.DetectSyntheticVideoRequest(video_file_data=chunk)

    def detect_synthetic_video(self, video_filepath):
        if not self.synthetic_key:
            return None
            
        metadata = (("authorization", f"Bearer {self.synthetic_key}"), ("function-id", self.synthetic_function_id))
        
        try:
            with grpc.secure_channel(self.target, credentials=grpc.ssl_channel_credentials()) as channel:
                stub = syntheticvideodetector_pb2_grpc.SyntheticVideoDetectorServiceStub(channel)
                responses = stub.DetectSyntheticVideo(
                    self._generate_synthetic_request(video_filepath),
                    metadata=metadata,
                )
                
                final_result = None
                for response in responses:
                    if response.HasField("final_result"):
                        final_result = response.final_result
                
                if final_result:
                    return {
                        "probability": final_result.probability,
                        "logit": final_result.logit,
                        "total_frames": final_result.total_clips
                    }
        except Exception as e:
            print(f"Synthetic Video Detector Error: {e}")
        return None

    def _generate_active_speaker_request(self, video_filepath):
        # We'll use embedded audio config
        video_config = video_pb2.VideoConfig(codec=video_pb2.VIDEO_CODEC_H264)
        detection_config = activespeakerdetection_pb2.ActiveSpeakerDetectionConfig(
            input_video_config=video_config,
            audio_source_config=activespeakerdetection_pb2.AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO
        )
        yield activespeakerdetection_pb2.DetectActiveSpeakerRequest(config=detection_config)
        
        with open(video_filepath, "rb") as f:
            while True:
                chunk = f.read(DATA_CHUNK_SIZE)
                if not chunk:
                    break
                data = activespeakerdetection_pb2.ActiveSpeakerDetectionData(video_data=chunk)
                yield activespeakerdetection_pb2.DetectActiveSpeakerRequest(data=data)

    def detect_active_speaker(self, video_filepath):
        if not self.active_speaker_key:
            return None
            
        metadata = (("authorization", f"Bearer {self.active_speaker_key}"), ("function-id", self.active_speaker_function_id))
        
        try:
            with grpc.secure_channel(self.target, credentials=grpc.ssl_channel_credentials()) as channel:
                stub = activespeakerdetection_pb2_grpc.ActiveSpeakerDetectionServiceStub(channel)
                responses = stub.DetectActiveSpeaker(
                    self._generate_active_speaker_request(video_filepath),
                    metadata=metadata,
                )
                
                frame_detections = {}
                for response in responses:
                    if response.HasField("active_speaker_detection_result"):
                        res = response.active_speaker_detection_result
                        speakers = []
                        for s in res.speaker_data:
                            speakers.append({
                                "is_speaking": s.is_speaking,
                                "confidence": s.face_detection_confidence,
                            })
                        frame_detections[res.frame_id] = speakers
                
                # Analyze active speaking frames
                total_speaking = 0
                for fid, spks in frame_detections.items():
                    if any(s["is_speaking"] for s in spks):
                        total_speaking += 1
                        
                return {
                    "total_frames": len(frame_detections),
                    "speaking_frames": total_speaking
                }
        except Exception as e:
            print(f"Active Speaker Detector Error: {e}")
        return None

    def detect_whisper_transcription(self, video_filepath):
        if not self.whisper_key:
            return None
            
        import requests
        url = "https://integrate.api.nvidia.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.whisper_key}"
        }
        
        try:
            with open(video_filepath, "rb") as f:
                files = {
                    "file": (os.path.basename(video_filepath), f, "audio/mp4")
                }
                data = {
                    "model": "nvidia/whisper-large-v3",
                    "response_format": "json"
                }
                response = requests.post(url, headers=headers, files=files, data=data)
                
            if response.status_code == 200:
                res_json = response.json()
                return {
                    "text": res_json.get("text", "")
                }
            else:
                print(f"Whisper API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Whisper API Error: {e}")
        return None

    def detect_nemotron_vl(self, video_filepath):
        if not self.nemotron_key:
            return None
        
        # Placeholder for Llama Nemotron Embed VL 1B integration
        # Multimodal embedding models typically use /v1/embeddings with base64 image data
        return {"status": "Integration pending multimodal payload specification"}

    def run_all(self, video_filepath):
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_synth = executor.submit(self.detect_synthetic_video, video_filepath)
            fut_speak = executor.submit(self.detect_active_speaker, video_filepath)
            fut_whisp = executor.submit(self.detect_whisper_transcription, video_filepath)
            fut_nemo = executor.submit(self.detect_nemotron_vl, video_filepath)
            
            results["synthetic_video"] = fut_synth.result()
            results["active_speaker"] = fut_speak.result()
            results["whisper"] = fut_whisp.result()
            results["nemotron"] = fut_nemo.result()
            
        return results

