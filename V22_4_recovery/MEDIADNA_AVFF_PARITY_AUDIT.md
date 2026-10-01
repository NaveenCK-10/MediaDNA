# MediaDNA AVFF Checkpoint & Preprocessing Parity Audit

## Track A: Checkpoint Integrity
The existing V14 AVFF checkpoint (`best_audio_model.pth`) was subjected to a strict programmatic audit against the `VideoCAVMAEFT` model architecture.
- **SHA256**: `108dccfc4a29ab0a59fb31b60c11aa928b08641c2a1c58db0239d8027c5531a9`
- **Model Parameter Count**: 187,015,458
- **Checkpoint Tensor Count**: 313
- **Model Tensor Count**: 313
- **Matched Keys**: 313
- **Missing Keys**: 0
- **Unexpected Keys**: 0
- **Shape Mismatches**: 0

**Conclusion**: The AVFF baseline architecture perfectly aligns with the `v14_fullscale` checkpoint. There is no missing state, and `strict=False` loading masks no architectural defects when applied to `VideoCAVMAEFT`.

## Track B: Preprocessing Parity
A line-by-line comparison between the training implementation (`tools/v22_3B_visual_train.py`, `tools/v22_3C_audio_train.py`) and the inference preprocessing reveals complete parity.

### Visual Modality
- **Sampling**: 16 frames uniformly extracted using `decord.VideoReader` via `np.linspace`.
- **Dimensions**: Extracted at exactly 224x224 (no separate crop/resize step required, preserving identical interpolation).
- **Normalization**: Standard ImageNet `[0.485, 0.456, 0.406]` applied sequentially.
- **Tensor Layout**: `(B, 3, 16, 224, 224)` permutation is maintained.

### Audio Modality
- **Extraction**: `ffmpeg` extracts exactly `pcm_s16le` at 16000Hz, single-channel.
- **MelSpectrogram**: Standard `torchaudio.transforms.MelSpectrogram` with `n_fft=1024, hop_length=160, n_mels=128`.
- **Padding/Clipping**: Spectrograms are hard-padded or clipped to exactly `target_len = 1024`.
- **Tensor Layout**: Permuted to `(B, 1024, 128)` matching the `AudioSpecialist` and AVFF requirements.

**Conclusion**: Inference preprocessing is mathematically identical to training preprocessing. The DEV ROC-AUC of 0.897 achieved by the AVFF baseline is not inflated or deflated by preprocessing discrepancies.
