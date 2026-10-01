"""
Module D: Temporal Forensic Branch

Provides a dedicated, independent temporal forensic analysis of a video.
Extracts frame-by-frame face crops, processes them through a lightweight 
frame encoder (MobileNetV2), and computes sequential temporal discontinuities
to identify deepfake flickering or temporal blending artifacts.
"""

import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms as T
from PIL import Image
from decord import VideoReader

class TemporalForensicsModule:
    def __init__(self, num_frames=16, im_res=224, device=None):
        """
        Initializes the Temporal Forensics Module.
        
        Args:
            num_frames: Number of frames to extract (matching standard sampling).
            im_res: Resolution for the face crop.
        """
        self.num_frames = num_frames
        self.im_res = im_res
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load OpenCV Haar Cascade for Face Tracking
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load Lightweight Frame Encoder (MobileNetV2)
        # Using a frozen pre-trained CNN to get robust facial features
        self.encoder = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Global Average Pooling to flatten the feature map
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(self.im_res, self.im_res)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_frames(self, video_path):
        """Uniformly samples frames to maintain sequence order."""
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            frames = [vr[i].asnumpy() for i in frame_indices]
            # Convert timestamps (assume 30fps for estimation if exact timing isn't easily accessible)
            timestamps = [idx / vr.get_avg_fps() for idx in frame_indices]
            return frames, timestamps, frame_indices
        except Exception as e:
            print(f"[TemporalForensics] Error reading video {video_path}: {e}")
            return [], [], []

    def _track_faces(self, frames):
        """
        Detects and tracks the primary face across the sequence.
        Returns face crops.
        """
        face_crops = []
        last_bbox = None
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            x, y, w, h = None, None, None, None
            if len(faces) > 0:
                # Take largest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                last_bbox = (x, y, w, h)
            elif last_bbox is not None:
                # Use last known bounding box (simple tracking for occlusion/misses)
                x, y, w, h = last_bbox
            else:
                # No face found, take center crop as fallback
                h_img, w_img = frame.shape[:2]
                w = h = min(h_img, w_img) // 2
                x = (w_img - w) // 2
                y = (h_img - h) // 2
                last_bbox = (x, y, w, h)
                
            # Crop
            crop = frame[y:y+h, x:x+w]
            face_crops.append(crop)
            
        return face_crops

    def extract_temporal_embeddings(self, face_crops):
        """
        Passes face crops through the frozen encoder to get a sequence of features.
        """
        batch = torch.stack([self.preprocess(crop) for crop in face_crops]).to(self.device)
        
        with torch.no_grad():
            features = self.encoder(batch)
            embeddings = self.pool(features).flatten(1) # Shape: (T, 1280)
            
        return embeddings

    def analyze(self, video_path):
        """
        Full temporal analysis pipeline.
        Computes frame-to-frame shifts and identifies temporal discontinuity.
        """
        frames, timestamps, frame_indices = self._get_frames(video_path)
        if len(frames) == 0:
            raise ValueError("No frames extracted.")
            
        # 1. Track Face Sequence
        face_crops = self._track_faces(frames)
        
        # 2. Extract Per-Frame Embeddings
        embeddings = self.extract_temporal_embeddings(face_crops) # (T, D)
        
        # 3. Compute Temporal Discontinuities (Frame-to-Frame L2 Distances)
        # Deepfakes often struggle with temporal consistency, leading to abrupt feature shifts
        shifts = []
        for i in range(1, len(embeddings)):
            dist = torch.norm(embeddings[i] - embeddings[i-1], p=2).item()
            shifts.append(dist)
            
        shifts = np.array(shifts)
        
        # 4. Statistical Temporal Features
        mean_shift = float(np.mean(shifts))
        max_shift = float(np.max(shifts))
        shift_variance = float(np.var(shifts))
        
        # Temporal Anomaly Score: highly localized spikes (max_shift relative to mean)
        # Fake videos often have localized flickering, causing high max_shift and high variance
        anomaly_score = max_shift * shift_variance
        
        # 5. Temporal Localization (Identify the highest jump)
        suspicious_intervals = []
        if len(shifts) > 0:
            peak_idx = np.argmax(shifts) + 1 # +1 because shifts is diffs
            t_start = timestamps[peak_idx - 1]
            t_end = timestamps[peak_idx]
            suspicious_intervals.append({
                "interval": f"{t_start:.2f}s - {t_end:.2f}s",
                "frame_indices": f"{frame_indices[peak_idx - 1]} - {frame_indices[peak_idx]}",
                "shift_magnitude": float(shifts[peak_idx - 1])
            })
            
        return {
            "temporal_anomaly_score": anomaly_score,
            "metrics": {
                "mean_shift": mean_shift,
                "max_shift": max_shift,
                "shift_variance": shift_variance
            },
            "temporal_evidence": suspicious_intervals,
            "temporal_model_used": "MobileNetV2 (Frozen Frame Encoder) + L2 Temporal Shift"
        }
