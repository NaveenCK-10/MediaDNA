# Scientific Limitations & Forensic Constraints

MediaDNA is an experimental forensic instrument. It is NOT a definitive oracle.

## 1. Dataset Dependence
The system is anchored entirely to the `FakeAVCeleb_v1.2` dataset distribution.

## 2. Modality Imbalance
The current V22.3/V22.4 pipeline shows an extreme dependency on the audio channel. Silent videos or aggressively compressed audio may yield unexpected or unreliable visual-only outputs.

## 3. Threshold Calibration
Calibration was fit strictly against `V22_2` predictions. The 0.95 (AUTHENTIC) / 0.96 (SYNTHETIC) boundaries are not universal truths.

## 4. OOD (Out of Distribution)
Generalization to wild deepfakes (e.g., Sora, Midjourney V6 video, advanced 2026 audio cloning) is unverified.

## 5. Security Limitations
MediaDNA currently lacks hardened sandboxing for FFmpeg parsing. Malicious video streams could exploit underlying `libav` vulnerabilities.
