import os
import sys
import time
import csv
import torch
import torchaudio
import numpy as np
import soundfile as sf
import torchvision.transforms as T
from decord import VideoReader
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.video_cav_mae import VideoCAVMAEFT

# PREPROCESSING CONFIG
FFMPEG_PATH = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AUDIO_SAMPLE_RATE = 16000
TARGET_LENGTH = 1024
NUM_MEL_BINS = 128
IM_RES = 224
NUM_FRAMES = 16
DATASET_MEAN = -5.081
DATASET_STD = 4.4849
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModalityAblationBenchmark:
    def __init__(self, checkpoint_path):
        self.model = VideoCAVMAEFT()
        self.model = torch.nn.DataParallel(self.model)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(DEVICE)
        self.model.eval()

    def _extract_audio(self, video_path):
        import io
        cmd = [FFMPEG_PATH, "-y", "-loglevel", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "wav", "pipe:1"]
        result = subprocess.run(cmd, check=True, capture_output=True)
        waveform, sr = sf.read(io.BytesIO(result.stdout))
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1, 2), size=(TARGET_LENGTH,), mode="linear", align_corners=False).transpose(1, 2).squeeze(0)
        return (fbank - DATASET_MEAN) / DATASET_STD

    def _extract_video(self, video_path):
        vr = VideoReader(video_path)
        frame_indices = np.linspace(0, len(vr) - 1, NUM_FRAMES).astype(int)
        frames = [vr[i].asnumpy() for i in frame_indices]
        tf = T.Compose([T.ToPILImage(), T.Resize((IM_RES, IM_RES)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return torch.stack([tf(f) for f in frames]).permute(1, 0, 2, 3)

    def infer(self, a_input, v_input):
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = self.model(a_input, v_input)
        return float(torch.sigmoid(output).cpu().float().numpy()[0][0])

def run_ablation():
    print("V22 MODALITY ABLATION EXPERIMENT")
    
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    benchmark = ModalityAblationBenchmark(ckpt_path)
    
    manifest_path = "V21_4_LOCKED_TEST_MANIFEST.csv"
    manifest = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest.append(row)
            
    results = []
    for item in manifest:
        try:
            fbank = benchmark._extract_audio(item["path"])
            frames = benchmark._extract_video(item["path"])
            
            a_input = fbank.unsqueeze(0).to(DEVICE)
            v_input = frames.unsqueeze(0).to(DEVICE)
            
            # FULL
            full_score = benchmark.infer(a_input, v_input)
            
            # AUDIO ONLY (Zero Video)
            zero_v = torch.zeros_like(frames).unsqueeze(0).to(DEVICE)
            audio_only_score = benchmark.infer(a_input, zero_v)
            
            # VIDEO ONLY (Zero Audio)
            zero_a = torch.zeros_like(fbank).unsqueeze(0).to(DEVICE)
            video_only_score = benchmark.infer(zero_a, v_input)
            
            results.append({
                "sample_id": item["sample_id"],
                "path": item["path"],
                "ground_truth": item["ground_truth"],
                "video_category": item["video_manipulation"],
                "full_score": full_score,
                "audio_only_score": audio_only_score,
                "video_only_score": video_only_score,
                "delta_audio": full_score - audio_only_score,
                "delta_video": full_score - video_only_score
            })
            print(f"Sample: {item['sample_id']} | Full: {full_score:.4f} | A-Only: {audio_only_score:.4f} | V-Only: {video_only_score:.4f}")
        except Exception as e:
            print(f"Error on {item['sample_id']}: {e}")
            
    out_file = "V22_MODALITY_ABLATION.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved Modality Ablation to {out_file}")

if __name__ == "__main__":
    run_ablation()
