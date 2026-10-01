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

class AVFFParityBenchmark:
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
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1, 2), size=(TARGET_LENGTH,), mode="linear", align_corners=False).transpose(1, 2).squeeze(0)
        return (fbank - DATASET_MEAN) / DATASET_STD

    def _extract_video(self, video_path):
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        # AVFF Parity: 5 fps, 16 frames = 3.2 seconds
        # So we want frames spaced by (fps / 5)
        step = max(1, int(fps / 5))
        frame_indices = np.arange(0, len(vr), step)[:NUM_FRAMES]
        # Pad if video is too short
        if len(frame_indices) < NUM_FRAMES:
            frame_indices = np.pad(frame_indices, (0, NUM_FRAMES - len(frame_indices)), mode='edge')
        
        frames = [vr[int(i)].asnumpy() for i in frame_indices]
        
        # AVFF Parity: Face Cropping approximation (Center Crop 50% to isolate face)
        # Assuming faces are generally centered in FakeAVCeleb
        tf = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(int(frames[0].shape[0] * 0.7)), # Crop center 70% to drop background
            T.Resize((IM_RES, IM_RES)), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return torch.stack([tf(f) for f in frames]).permute(1, 0, 2, 3)

    def infer(self, video_path):
        fbank = self._extract_audio(video_path)
        frames = self._extract_video(video_path)
        
        a_input = fbank.unsqueeze(0).to(DEVICE)
        v_input = frames.unsqueeze(0).to(DEVICE)
        
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = self.model(a_input, v_input)
                
        # Also run visual-only to see if visual pathway improved
        zero_a = torch.zeros_like(fbank).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                v_only = self.model(zero_a, v_input)
                
        return float(torch.sigmoid(output).cpu().float().numpy()[0][0]), float(torch.sigmoid(v_only).cpu().float().numpy()[0][0])

def main():
    print("V22 AVFF PREPROCESSING PARITY EXPERIMENT")
    
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "v14_fullscale", "models", "best_audio_model.pth")
    benchmark = AVFFParityBenchmark(ckpt_path)
    
    manifest_path = "V21_4_LOCKED_TEST_MANIFEST.csv"
    manifest = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest.append(row)
            
    results = []
    for item in manifest:
        try:
            full_score, v_only = benchmark.infer(item['path'])
            results.append({
                "sample_id": item["sample_id"],
                "path": item["path"],
                "ground_truth": item["ground_truth"],
                "video_category": item["video_manipulation"],
                "parity_full_score": full_score,
                "parity_v_only_score": v_only
            })
            print(f"Sample: {item['sample_id']} | Full: {full_score:.4f} | V-Only (Parity): {v_only:.4f}")
        except Exception as e:
            print(f"Error on {item['sample_id']}: {e}")
            
    out_file = "V22_AVFF_PARITY_RESULTS.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSaved AVFF Parity Results to {out_file}")
    
    with open("V22_PREPROCESSING_PARITY.md", "a") as f:
        f.write("\n\n### Execution Results\n")
        f.write("The parity execution evaluated 5 FPS, 16 frames (3.2s) with Center Cropping (Face proxy).\n")
        f.write("Results indicate whether this specific preprocessing can reactivate the visual branch (which previously output ~0.54 universally).\n")

if __name__ == "__main__":
    main()
