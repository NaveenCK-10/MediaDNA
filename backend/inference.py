"""
OpenAVFF Inference Service for MediaDNA.

This module wraps the existing OpenAVFF VideoCAVMAEFT model for single-video
inference. It reuses the EXACT same preprocessing logic from src/dataloader.py
(VideoAudioEvalDataset) to ensure CLI-GUI consistency.

Key design decisions:
- Model loaded ONCE at startup, kept in GPU memory
- model.eval() always called (fixes eval.py omission where dropout was active)
- torch.no_grad() + autocast() used for inference
- Preprocessing constants match eval.py exactly
- Temporary audio files cleaned up after each inference
"""
import os
import sys
import time
import tempfile
import subprocess
import logging

import torch
import torchaudio
import numpy as np
import soundfile as sf
import torchvision.transforms as T
from torch.cuda.amp import autocast
from decord import VideoReader
from dataclasses import dataclass, field
from typing import List, Optional

# Add the project root to sys.path so we can import src.models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

logger = logging.getLogger("mediadna.inference")

# ─── Constants (from eval.py / dataloader.py, verified in audit) ─────────────
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
DATASET_MEAN = -5.081
DATASET_STD = 4.4849
TARGET_LENGTH = 1024
NUM_MEL_BINS = 128
IM_RES = 224
NUM_FRAMES = 16
AUDIO_SAMPLE_RATE = 16000

# ImageNet normalization (from dataloader.py lines 377-379)
IMAGENET_MEAN = [0.4850, 0.4560, 0.4060]
IMAGENET_STD = [0.2290, 0.2240, 0.2250]


@dataclass
class AnalysisResult:
    """Result of a single video analysis."""
    prediction: str  # "fake" or "real"
    fake_probability: float
    real_probability: float
    raw_logits: List[float] = field(default_factory=list)
    inference_time: float = 0.0  # model inference only
    total_time: float = 0.0  # includes preprocessing
    frames_processed: int = 0
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    video_filename: str = ""


class OpenAVFFService:
    """
    Wraps the OpenAVFF VideoCAVMAEFT model for single-video inference.
    
    Model is loaded once and kept in memory. All preprocessing replicates
    the exact logic from src/dataloader.py VideoAudioEvalDataset.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the service and load the model.
        
        Args:
            checkpoint_path: Path to the trained checkpoint (.pth file).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.total_params = 0
        
        # Use V14 full-scale checkpoint by default
        self.default_checkpoint = os.path.join(
            PROJECT_ROOT, 
            "checkpoints", 
            "v14_fullscale", 
            "models", 
            "best_audio_model.pth"
        )
        
        self.checkpoint_path = os.path.abspath(checkpoint_path or self.default_checkpoint)

        # Visual preprocessing — matches VideoAudioEvalDataset
        # FIXED: Use Resize instead of CenterCrop to match training.
        self._visual_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IM_RES, IM_RES)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._load_model()

    def _load_model(self):
        """
        Load the VideoCAVMAEFT model from checkpoint.
        
        Replicates the exact loading from eval.py:
        1. VideoCAVMAEFT() → DataParallel → load_state_dict
        2. Assert no missing/unexpected keys
        3. Move to device
        4. Set to eval mode (fixes eval.py bug)
        """
        logger.info(f"Loading model from: {self.checkpoint_path}")
        logger.info(f"Device: {self.device}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Create model — same as eval.py L19-20
        self.model = VideoCAVMAEFT()
        self.model = torch.nn.DataParallel(self.model)

        # Load checkpoint — same as eval.py L21-23
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        miss, unexp = self.model.load_state_dict(ckpt, strict=False)
        
        if len(miss) > 0 or len(unexp) > 0:
            logger.warning(f"Checkpoint load: {len(miss)} missing, {len(unexp)} unexpected keys")
            if len(miss) > 0:
                logger.warning(f"Missing: {miss}")
            if len(unexp) > 0:
                logger.warning(f"Unexpected: {unexp}")
        
        assert len(miss) == 0 and len(unexp) == 0, (
            f"Checkpoint mismatch! Missing: {len(miss)}, Unexpected: {len(unexp)}"
        )

        self.total_params = sum(p.numel() for p in self.model.parameters())

        self.model.to(self.device)
        self.model.eval()  # IMPORTANT: eval.py omits this — we fix it here
        
        # Optimize CUDNN
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        logger.info(f"Model loaded successfully. Parameters: {self.total_params:,}")
        logger.info(f"Model is in eval mode: {not self.model.training}")

    def _extract_audio_fbank(self, video_path: str) -> torch.Tensor:
        """
        Extract audio from video and compute filterbank features.
        
        Replicates VideoAudioEvalDataset._wav2fbank() exactly (dataloader.py L382-418):
        1. FFmpeg → WAV (16kHz, mono)
        2. soundfile.read → waveform
        3. Mean subtraction
        4. Kaldi fbank (128 mel bins, hanning window, htk_compat)
        5. Interpolate to target_length (1024)
        6. Normalize with dataset mean/std
        
        Returns:
            Normalized fbank tensor of shape [1024, 128]
        """
        import io

        try:
            # FFmpeg extraction directly to stdout (WAV format)
            cmd = [
                FFMPEG_PATH, "-y", "-loglevel", "error",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
                "-f", "wav", "pipe:1",
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True)
            
            # Read directly from memory buffer
            waveform, sr = sf.read(io.BytesIO(result.stdout))
            waveform = torch.tensor(waveform).unsqueeze(0).float()
            waveform = waveform - waveform.mean()

            # Kaldi fbank
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=NUM_MEL_BINS,
                dither=0.0,
                frame_shift=10,
            )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg audio extraction failed for {video_path}: {e.stderr.decode('utf-8')}"
            )
        except Exception as e:
            raise RuntimeError(f"Audio processing failed for {video_path}: {e}")

        # Interpolate to target_length — same as dataloader.py L416
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2),
            size=(TARGET_LENGTH,),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).squeeze(0)

        # Normalize — same as dataloader.py L462-463
        fbank = (fbank - DATASET_MEAN) / DATASET_STD

        return fbank

    def _extract_video_frames(self, video_path: str) -> tuple:
        """
        Extract and preprocess video frames.
        
        Replicates VideoAudioEvalDataset._get_frames() + preprocessing (dataloader.py L420-478):
        1. decord VideoReader → sample 16 uniform frames
        2. CenterCrop(224) → ToTensor → ImageNet normalize
        3. Stack → permute to (C, T, H, W)
        
        Returns:
            Tuple of (frames_tensor [C, T, H, W], num_frames_processed)
        """
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)

            # Uniform sampling — same as dataloader.py L426
            frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)

            # Read frames — same as dataloader.py L429
            frames = [vr[i].asnumpy() for i in frame_indices]
            num_frames = len(frames)

        except Exception as e:
            raise RuntimeError(f"Video frame extraction failed for {video_path}: {e}")

        # Preprocess — same as dataloader.py L446-447
        frames = [self._visual_transform(frame) for frame in frames]
        frames = torch.stack(frames)

        # Permute — same as dataloader.py L478: (T, C, H, W) → (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)

        return frames, num_frames

    def analyze_video(self, video_path: str) -> AnalysisResult:
        """
        Run full inference on a single video file.
        
        This is the main entry point. Replicates the eval.py inference loop
        for a single video:
        1. Extract audio → fbank tensor
        2. Extract video → frame tensor
        3. Model forward pass with autocast
        4. sigmoid → probabilities
        5. Threshold at 0.5
        
        Args:
            video_path: Absolute path to the video file.
            
        Returns:
            AnalysisResult with prediction, probabilities, timing, etc.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        total_start = time.time()
        video_filename = os.path.basename(video_path)

        # Step 1: Extract and preprocess audio
        logger.info(f"Extracting audio from: {video_filename}")
        fbank = self._extract_audio_fbank(video_path)

        # Step 2: Extract and preprocess video frames
        logger.info(f"Extracting frames from: {video_filename}")
        frames, num_frames = self._extract_video_frames(video_path)

        # Step 3: Prepare tensors — add batch dimension
        a_input = fbank.unsqueeze(0).to(self.device)     # [1, 1024, 128]
        v_input = frames.unsqueeze(0).to(self.device)    # [1, 3, 16, 224, 224]

        # Step 4: Model inference — same as eval.py L46-53
        logger.info("Running model inference...")
        inference_start = time.time()

        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = self.model(a_input, v_input)
                
        # Free memory aggressively after forward pass
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Step 5: Compute probabilities — same as eval.py L53
        # sigmoid(output) where output[0] = Fake logit, output[1] = Real logit
        probabilities = torch.sigmoid(output).cpu().float().numpy()[0]
        raw_logits = output.cpu().float().numpy()[0].tolist()

        inference_time = time.time() - inference_start
        total_time = time.time() - total_start

        # Step 6: Extract P(Fake) — same as eval.py L56
        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])

        # Step 7: Threshold at 0.60 based on V15.3 Validation Sweep
        prediction = "fake" if fake_prob >= 0.60 else "real"

        logger.info(
            f"Result: {prediction} (P(Fake)={fake_prob:.4f}, P(Real)={real_prob:.4f}) "
            f"in {inference_time:.2f}s"
        )

        return AnalysisResult(
            prediction=prediction,
            fake_probability=fake_prob,
            real_probability=real_prob,
            raw_logits=raw_logits,
            inference_time=round(inference_time, 3),
            total_time=round(total_time, 3),
            frames_processed=num_frames,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            video_filename=video_filename,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(0)
        return "CPU"

    @property
    def gpu_memory_mb(self) -> Optional[int]:
        if self.device.type == "cuda":
            return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        return None
