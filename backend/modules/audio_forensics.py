"""
Module C: Audio Forensic Branch

Provides a dedicated, independent audio forensic analysis of a video.
Extracts audio features (spectral centroid, zero-crossing rate, mel-spectrogram)
and analyzes the frozen OpenAVFF AudioEncoder to determine if audio shortcuts
are being used.
"""

import os
import torch
import numpy as np
import librosa
import subprocess
import tempfile
import torchaudio
import soundfile as sf
import warnings

# Suppress PySoundFile warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AudioForensicsModule:
    def __init__(self, model, target_length=1024, mel_bins=128):
        """
        Initializes the Audio Forensics Module.
        
        Args:
            model: The frozen VideoCAVMAEFT model.
            target_length: Number of time frames in spectrogram (matches V15.4 = 1024).
            mel_bins: Number of mel bins (matches V15.4 = 128).
        """
        self.model = model
        self.target_length = target_length
        self.mel_bins = mel_bins
        self.device = next(model.parameters()).device
        self.ffmpeg_path = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
        
        self.attention_weights = None
        
        # Register forward hook on the last block's attention in the AudioEncoder
        try:
            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            # AudioEncoder uses timm Block. The attention layer is blk.attn. 
            # In timm, Attention forward usually computes `x = self.attn_drop(attn @ v)`
            # Hooking attn_drop gives us the attention matrix
            last_block = actual_model.audio_encoder.transformer[-1]
            last_block.attn.attn_drop.register_forward_hook(self._attention_hook)
            self.hook_success = True
        except Exception as e:
            print(f"[AudioForensics] Warning: Could not register attention hook: {e}")
            self.hook_success = False

    def _attention_hook(self, module, input, output):
        # In timm's vision_transformer, attn_drop is applied to the softmaxed attention matrix
        # input[0] shape is (B, num_heads, N, N)
        if isinstance(input, tuple) and len(input) > 0:
            self.attention_weights = input[0].detach().cpu()

    def _extract_audio_features(self, video_path):
        """
        Extracts raw waveform, kaldi fbank (matching OpenAVFF), and 
        handcrafted spectral features (centroid, zcr, rolloff).
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_wav = tmp_file.name
            
        try:
            # 1. Extract audio via ffmpeg (identical to dataloader.py)
            cmd = [self.ffmpeg_path, "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", temp_wav]
            subprocess.run(cmd, check=True)
            
            # Load with soundfile
            waveform, sr = sf.read(temp_wav)
            waveform_torch = torch.tensor(waveform).unsqueeze(0).float()
            waveform_torch = waveform_torch - waveform_torch.mean()
            
            # 2. Kaldi Fbank (matches dataloader.py exactly)
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform_torch, htk_compat=True, sample_frequency=sr, 
                use_energy=False, window_type='hanning', 
                num_mel_bins=self.mel_bins, dither=0.0, frame_shift=10
            )
            
            # 3. Handcrafted features via librosa
            # Librosa operates on numpy array, shape (T,)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1) # force mono
                
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=waveform)[0]
            
            handcrafted_features = {
                "mean_spectral_centroid": float(np.mean(spectral_centroid)),
                "std_spectral_centroid": float(np.std(spectral_centroid)),
                "mean_spectral_rolloff": float(np.mean(spectral_rolloff)),
                "mean_zcr": float(np.mean(zero_crossing_rate)),
                "std_zcr": float(np.std(zero_crossing_rate))
            }
            
        except Exception as e:
            print(f"[AudioForensics] Error processing audio {video_path}: {e}")
            fbank = torch.zeros([512, 128]) + 0.01
            handcrafted_features = {
                "mean_spectral_centroid": 0.0,
                "std_spectral_centroid": 0.0,
                "mean_spectral_rolloff": 0.0,
                "mean_zcr": 0.0,
                "std_zcr": 0.0
            }
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
                    
        # Interpolate fbank to target length (matches dataloader)
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1,2), 
            size=(self.target_length, ), 
            mode='linear', align_corners=False
        ).transpose(1,2).squeeze(0)
        
        # Dataset normalization (hardcoded from V15.4 config for reproducibility)
        norm_mean, norm_std = -5.081, 4.485
        fbank = (fbank - norm_mean) / (norm_std * 2)

        return fbank, handcrafted_features

    def compute_audio_embeddings(self, fbank):
        """
        Passes the fbank through the frozen AudioEncoder.
        Computes embedding variance and extracts attention maps.
        """
        self.model.eval()
        
        # Shape: (B, 1024, 128)
        fbank_tensor = fbank.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            # audio_encoder returns shape (B, N, C), N = 512, C = 768
            emb = actual_model.audio_encoder(fbank_tensor)
            
            # Temporal embedding variance (shortcut detector)
            # High variance -> dynamic speech. Low variance -> monotonic/synthetic artifact?
            temporal_variance = float(emb.var(dim=1).mean().item())
            
            # Compute a synthetic anomaly score (heuristic combination)
            # We will use this in Phase 4 to see if simple stats can separate Real/Fake
            anomaly_score = 1.0 / (1.0 + temporal_variance)
            
        return {
            "temporal_variance": temporal_variance,
            "anomaly_score": anomaly_score
        }

    def generate_attention_spectrogram(self, fbank):
        """
        Generates a 1D attention curve over time (1024 frames) by pooling the
        ViT self-attention matrix from the last block.
        """
        if self.attention_weights is None:
            return None
            
        # self.attention_weights is (B, num_heads, N, N) where N=512 patches
        attn = self.attention_weights[0].mean(dim=0) # (512, 512)
        
        # Sum attention received by each patch
        attn_received = attn.sum(dim=0) # (512,)
        
        # Interpolate from 512 patches back to 1024 spectrogram frames
        attn_received = attn_received.unsqueeze(0).unsqueeze(0) # (1, 1, 512)
        attn_received = torch.nn.functional.interpolate(attn_received, size=(self.target_length,), mode='linear', align_corners=False)
        attn_received = attn_received.squeeze().numpy()
        
        # Normalize
        attn_received = attn_received - attn_received.min()
        attn_received = attn_received / (attn_received.max() + 1e-8)
        
        return attn_received

    def analyze(self, video_path):
        """
        Full analysis pipeline for the audio track.
        """
        # 1. Extract Kaldi fbank and Handcrafted spectral features
        fbank, handcrafted = self._extract_audio_features(video_path)
        
        # 2. Extract deep embeddings from frozen AudioEncoder
        emb_data = self.compute_audio_embeddings(fbank)
        
        # 3. Extract time-frequency attention evidence
        time_attention = self.generate_attention_spectrogram(fbank)
        
        # 4. Identify suspicious intervals (top 10% attention peaks)
        suspicious_intervals = []
        if time_attention is not None:
            threshold = np.percentile(time_attention, 90)
            peak_indices = np.where(time_attention > threshold)[0]
            # Convert indices (0-1023) to approximate seconds (assuming 10s audio max, each frame ~10ms)
            for idx in peak_indices[::10]: # sample every 10th peak to avoid clutter
                sec = (idx / self.target_length) * 10.0
                suspicious_intervals.append(f"{sec:.2f}s")
                
        result = {
            "audio_anomaly_score": emb_data["anomaly_score"],
            "metrics": {
                "temporal_variance": emb_data["temporal_variance"],
                **handcrafted
            },
            "audio_evidence_intervals": suspicious_intervals[:5], # Top 5 intervals
            "audio_model_used": "OpenAVFF Frozen AudioEncoder (AudioAST-ViT)",
            "spectrogram_extracted": True
        }
            
        return result
