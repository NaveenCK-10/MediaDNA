# Reproducibility Guide

## Environment
Ensure PyTorch 2.0+ with CUDA 11.8.

## Checkpoint Identity
The baseline uses `best_audio_model.pth` (V14). SHA256 integrity checks must pass the 313 tensor audit script.

## Splitting
MediaDNA relies on strict, immutable dataset manifesting. All samples in `locked_test` are cryptographically verified to have zero overlap with `train` and `dev` folds.

## Preprocessing Parity
- **Video:** 16 frames uniformly extracted (ffprobe duration / 16).
- **Audio:** 16kHz PCM, 1024 frames, 128 Mel bands.
