import os
import sys
import csv
import torch
import numpy as np
from decord import VideoReader
import torchvision.transforms as T
import torchaudio
import soundfile as sf
import io
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AUDIO_SAMPLE_RATE = 16000
TARGET_LENGTH = 1024
NUM_MEL_BINS = 128
IM_RES = 224
NUM_FRAMES = 16
DATASET_MEAN = -5.081
DATASET_STD = 4.4849
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValidAblationBenchmark:
    def __init__(self, checkpoint_path):
        self.model = VideoCAVMAEFT()
        self.model = torch.nn.DataParallel(self.model)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(DEVICE)
        self.model.eval()

    def _extract_audio(self, video_path):
        cmd = [FFMPEG_PATH, "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "wav", "pipe:1"]
        result = subprocess.run(cmd, check=True, capture_output=True)
        waveform, sr = sf.read(io.BytesIO(result.stdout))
        return self._process_waveform(waveform, sr)
        
    def _process_waveform(self, waveform, sr):
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1, 2), size=(TARGET_LENGTH,), mode="linear", align_corners=False).transpose(1, 2).squeeze(0)
        return (fbank - DATASET_MEAN) / DATASET_STD

    def _extract_video(self, video_path):
        vr = VideoReader(video_path)
        frame_indices = np.linspace(0, len(vr) - 1, NUM_FRAMES).astype(int)
        frames = [vr[i].asnumpy() for i in frame_indices]
        return self._process_frames(frames)
        
    def _process_frames(self, frames):
        tf = T.Compose([T.ToPILImage(), T.Resize((IM_RES, IM_RES)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return torch.stack([tf(f) for f in frames]).permute(1, 0, 2, 3)

    def infer(self, a_input, v_input):
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = self.model(a_input, v_input)
        return float(torch.sigmoid(output).cpu().float().numpy()[0][0])

def main():
    print("V22 VALID VISUAL ABLATION")
    
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    benchmark = ValidAblationBenchmark(ckpt_path)
    
    # 1. We test if zeroing out tensors is the same as black frames / silence.
    manifest_path = "V21_4_LOCKED_TEST_MANIFEST.csv"
    manifest = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest.append(row)
            
    sample = manifest[0]
    print(f"Testing on sample {sample['sample_id']} (Path: {sample['path']})")
    
    fbank = benchmark._extract_audio(sample['path'])
    frames = benchmark._extract_video(sample['path'])
    
    a_input = fbank.unsqueeze(0).to(DEVICE)
    v_input = frames.unsqueeze(0).to(DEVICE)
    
    full_score = benchmark.infer(a_input, v_input)
    
    # --- AUDIO ABLATION VALIDATION ---
    # Zero tensor (Previous method)
    zero_a = torch.zeros_like(fbank).unsqueeze(0).to(DEVICE)
    score_zero_a = benchmark.infer(zero_a, v_input)
    
    # Absolute Silence Waveform (True physical ablation)
    # Target length approx equal to standard
    silence_wav = np.zeros(16000 * 5) # 5 seconds of silence
    true_silence_fbank = benchmark._process_waveform(silence_wav, 16000).unsqueeze(0).to(DEVICE)
    score_true_silence = benchmark.infer(true_silence_fbank, v_input)
    
    # --- VISUAL ABLATION VALIDATION ---
    # Zero tensor (Previous method)
    zero_v = torch.zeros_like(frames).unsqueeze(0).to(DEVICE)
    score_zero_v = benchmark.infer(a_input, zero_v)
    
    # Absolute Black Frames (True physical ablation)
    black_frames = [np.zeros((IM_RES, IM_RES, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]
    true_black_v = benchmark._process_frames(black_frames).unsqueeze(0).to(DEVICE)
    score_true_black = benchmark.infer(a_input, true_black_v)
    
    print("\n--- ABLATION VALIDATION RESULTS ---")
    print(f"FULL SCORE: {full_score:.4f}")
    print(f"\n[AUDIO ABLATION]")
    print(f"  Zero Tensor Injection:  {score_zero_a:.4f}")
    print(f"  True Silence Waveform:  {score_true_silence:.4f}")
    print(f"  Difference:             {abs(score_zero_a - score_true_silence):.4f}")
    
    print(f"\n[VISUAL ABLATION]")
    print(f"  Zero Tensor Injection:  {score_zero_v:.4f}")
    print(f"  True Black Frames:      {score_true_black:.4f}")
    print(f"  Difference:             {abs(score_zero_v - score_true_black):.4f}")
    
    print("\nCONCLUSION: If True physical ablation significantly diverges from Zero Tensor injection, the previous visual ablation was INVALID because a zero tensor in a normalized feature space represents an out-of-distribution input, corrupting the model's cross-attention rather than purely dropping the modality.")

    with open("V22_ABLATION_VALIDATION_REPORT.md", "w") as f:
        f.write("# V22 ABLATION VALIDATION REPORT\n\n")
        f.write(f"**FULL SCORE**: {full_score:.4f}\n\n")
        f.write("### AUDIO ABLATION\n")
        f.write(f"- Zero Tensor: {score_zero_a:.4f}\n")
        f.write(f"- True Silence: {score_true_silence:.4f}\n")
        f.write(f"- Diff: {abs(score_zero_a - score_true_silence):.4f}\n\n")
        f.write("### VISUAL ABLATION\n")
        f.write(f"- Zero Tensor: {score_zero_v:.4f}\n")
        f.write(f"- True Black Frames: {score_true_black:.4f}\n")
        f.write(f"- Diff: {abs(score_zero_v - score_true_black):.4f}\n\n")
        if abs(score_zero_v - score_true_black) > 0.05:
            f.write("**CONCLUSION**: The visual ablation is INVALID. Zero-tensor injection creates an extreme out-of-distribution feature state due to mean/std normalization. Physical ablation (black frames run through normalization) must be used.")
        else:
            f.write("**CONCLUSION**: The visual ablation is VALID. Zero-tensor injection closely mirrors physical blanking.")
            
if __name__ == "__main__":
    main()
