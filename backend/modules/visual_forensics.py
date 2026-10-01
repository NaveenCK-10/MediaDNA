"""
Module B: Visual Forensic Branch

Provides a dedicated, independent visual forensic analysis of a video.
Extracts facial regions using OpenCV and genuine visual feature
measurements (spatial and temporal anomalies) from the frozen OpenAVFF model.
"""

import os
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image
from decord import VideoReader

class VisualForensicsModule:
    def __init__(self, model, num_frames=16, im_res=224):
        """
        Initializes the Visual Forensics Module.
        
        Args:
            model: The frozen VideoCAVMAEFT model.
            num_frames: Number of frames to extract (matches V15.4 baseline = 16).
            im_res: Resolution to resize frames (matches V15.4 baseline = 224).
        """
        self.model = model
        self.num_frames = num_frames
        self.im_res = im_res
        self.device = next(model.parameters()).device
        
        # Load OpenCV Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Preprocessing exactly matching V15.4
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(self.im_res, self.im_res)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Hook for capturing attention weights
        self.attention_weights = None
        
        # Register a forward hook on the last block's attention dropout layer.
        try:
            # Handle DataParallel wrapped models
            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            last_block = actual_model.visual_encoder.blocks[-1]
            last_block.attn.attn_drop.register_forward_hook(self._attention_hook)
            self.hook_success = True
        except Exception as e:
            print(f"[VisualForensics] Warning: Could not register attention hook: {e}")
            self.hook_success = False

    def _attention_hook(self, module, input, output):
        # input[0] is the softmax attention matrix (B, num_heads, N, N)
        self.attention_weights = input[0].detach().cpu()

    def _get_frames(self, video_path):
        """Uniformly samples frames matching the original dataloader."""
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            frames = [vr[i].asnumpy() for i in frame_indices]
            return frames
        except Exception as e:
            print(f"[VisualForensics] Error reading video: {e}")
            return [np.zeros((self.im_res, self.im_res, 3), dtype=np.uint8) for _ in range(self.num_frames)]

    def detect_faces(self, frames):
        """
        Detect faces using OpenCV Haar Cascades.
        Returns face metadata for each frame.
        """
        face_data = []
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            frame_info = {
                "frame_idx": i,
                "face_detected": False,
                "bbox": None,
                "confidence": 0.0
            }
            
            if len(faces) > 0:
                # Take largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                
                frame_info["face_detected"] = True
                frame_info["confidence"] = 1.0 # Haar doesn't provide confidence, use 1.0
                
                # Normalize bbox to [0, 1]
                img_h, img_w = frame.shape[:2]
                frame_info["bbox"] = [
                    float(x)/img_w, float(y)/img_h, 
                    float(w)/img_w, float(h)/img_h
                ]
            
            face_data.append(frame_info)
            
        return face_data

    def compute_visual_features(self, frames):
        """
        Pass frames through the visual encoder to extract features.
        Computes temporal inconsistency score.
        """
        # Preprocess matching V15.4
        tensor_frames = [self.preprocess(frame) for frame in frames]
        tensor_frames = torch.stack(tensor_frames)  # (T, C, H, W)
        tensor_frames = tensor_frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        tensor_frames = tensor_frames.to(self.device)
        
        # Extract features using the frozen VisualEncoder
        self.model.eval()
        with torch.no_grad():
            actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            # visual_encoder returns (B, 1568, 768)
            emb = actual_model.visual_encoder(tensor_frames)
            
            # Reshape to separate temporal and spatial dimensions
            # 1568 = 8 temporal tokens * 196 spatial tokens (14x14)
            # Since tubelet_size=2, 16 frames -> 8 temporal tokens
            B, N, C = emb.shape
            emb = emb.reshape(B, 8, 196, C)
            
            # Temporal Anomaly: Frame-to-frame mean absolute error in feature space
            # Represents structural/textural jitter across time
            temporal_diff = torch.abs(emb[:, 1:, :, :] - emb[:, :-1, :, :])
            temporal_anomaly = float(temporal_diff.mean().item())
            
            # Spatial Anomaly: Variation across spatial tokens
            spatial_variance = float(emb.var(dim=2).mean().item())
            
            # Aggregate Visual Anomaly Score (normalized to rough [0, 1] range)
            # These normalizers are derived heuristically; will be properly calibrated in Phase 9
            norm_temporal = min(temporal_anomaly / 2.0, 1.0)
            
        return {
            "temporal_feature_mae": temporal_anomaly,
            "spatial_variance": spatial_variance,
            "visual_anomaly_score": norm_temporal
        }

    def generate_attention_heatmap(self):
        """
        Generates an attention heatmap over the 14x14 spatial grid.
        Sums attention received by each spatial patch across all heads.
        """
        if self.attention_weights is None:
            return None
            
        # self.attention_weights is (B, num_heads, N, N) where N=1568
        # We want to see which patches receive the most attention.
        attn = self.attention_weights[0]  # Take first batch item (num_heads, 1568, 1568)
        
        # Average across heads -> (1568, 1568)
        attn_mean = attn.mean(dim=0)
        
        # Sum attention received by each patch (sum over rows/queries)
        attn_received = attn_mean.sum(dim=0)  # (1568,)
        
        # Reshape back to temporal and spatial
        attn_received = attn_received.reshape(8, 14, 14)
        
        # Average across temporal dimension to get a single 14x14 spatial heatmap
        spatial_heatmap = attn_received.mean(dim=0)  # (14, 14)
        
        # Normalize to [0, 1]
        spatial_heatmap = spatial_heatmap - spatial_heatmap.min()
        spatial_heatmap = spatial_heatmap / (spatial_heatmap.max() + 1e-8)
        
        return spatial_heatmap.numpy()

    def analyze(self, video_path):
        """
        Full analysis pipeline for a single video.
        """
        # 1. Extract frames
        frames = self._get_frames(video_path)
        
        # 2. Face detection
        face_data = self.detect_faces(frames)
        
        # 3. Visual feature extraction & anomaly scoring
        feature_data = self.compute_visual_features(frames)
        
        # 4. Extract attention heatmap
        heatmap = self.generate_attention_heatmap()
        
        # Format the output matching the requested API structure
        result = {
            "visual_anomaly_score": feature_data["visual_anomaly_score"],
            "frames_analyzed": self.num_frames,
            "faces_detected": sum(1 for f in face_data if f["face_detected"]),
            "frame_scores": face_data,  # Face detection metadata per frame
            "metrics": {
                "temporal_mae": feature_data["temporal_feature_mae"],
                "spatial_variance": feature_data["spatial_variance"]
            }
        }
        
        # In a real API, the heatmap would be saved as an image and the path returned
        # For evaluation, we return the array or None
        if heatmap is not None:
            result["heatmap_available"] = True
        else:
            result["heatmap_available"] = False
            
        return result
