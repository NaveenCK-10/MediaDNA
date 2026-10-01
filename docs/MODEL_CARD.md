# MediaDNA (OpenAVFF V16) Model Card

## Model Details
- **Architecture**: VideoCAVMAEFT (Cross-Attention Masked Auto-Encoder)
- **Task**: Deepfake Detection & Multimodal Provenance
- **Version**: V16.1

## Intended Use
Research demonstration of multimodal digital forensics.

## Metrics
- **Accuracy**: 68.1% (Balanced)
- **ROC-AUC**: 0.9231 (INVALIDATED - Used random proxy features)
- **Calibration**: Brier Score measured on validation data.

## Limitations
- Blindspot on high-quality visual manipulations with original audio (FakeVideo-RealAudio).
- Cannot identify specific generative AI tools (e.g. Midjourney, Wav2Lip) with high confidence.
