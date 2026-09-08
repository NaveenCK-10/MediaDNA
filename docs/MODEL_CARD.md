# MediaDNA (OpenAVFF) Model Card

## Model Details
*   **Architecture:** VideoCAVMAEFT (Cross-Attention Video-Audio Masked Auto-Encoder Fine-Tuned).
*   **Modality:** Multimodal (Audio-Visual).
*   **Parameters:** ~113.8M
*   **Framework:** PyTorch
*   **Input Specifications:** 
    *   Visual: 16 Frames, 224x224 RGB, ImageNet Normalized.
    *   Audio: 16kHz, 80-bin Kaldi Log-Mel Filterbank, 1024 temporal frames.
*   **Developer:** OpenAVFF Research Team / MediaDNA integration.

## Intended Use
**Primary Use Case:** Research and academic evaluation of cross-modal attention mechanisms in deepfake detection. 
**Out-of-Scope Use:** Automated moderation, legal forensics, evidence authentication, or production deployment. This model acts as a research benchmark and should not be used as a standalone authority for media authenticity.

## Training Data
*   **Dataset:** FakeAVCeleb (v1.2).
*   **Composition:** 
    *   Authentic YouTube videos of varied demographics.
    *   Synthetically manipulated videos utilizing face-swap (Faceswap, FSGAN) and audio cloning (SV2TTS, Wav2Lip).
*   **Pre-training:** The base encoder leverages pre-trained CAVMAE weights optimized on AudioSet and Kinetics-400.

## Evaluation Data
*   **Test Set:** 5,145 held-out samples from the FakeAVCeleb dataset, ensuring no identity overlap with the training set.

## Metrics & Performance
*   **Global Accuracy:** 64.69%
*   **ROC-AUC:** 0.6944
*   **RealVideo-FakeAudio (Audio Deepfake):** 96.43% Accuracy.
*   **FakeVideo-FakeAudio (Full Deepfake):** 95.20% Accuracy.

## Known Blindspots & Limitations
1.  **The "FakeVideo-RealAudio" Blindspot:** The model demonstrates a critical failure rate (27.20% accuracy) when assessing media where the visual stream is synthetic but the audio stream is perfectly authentic. The cross-attention mechanism over-indexes on acoustic features, causing it to dismiss visual artifacts if the audio sounds genuine.
2.  **Acoustic Noise Vulnerability:** The model possesses zero robustness against environmental audio noise. Applying a standard 10dB SNR Gaussian noise profile completely destabilizes the network, resulting in catastrophic false-positive "Fake" classifications.
3.  **Low-Resolution Resilience:** The model maintains reasonable accuracy on heavily compressed (CRF 40) or low-resolution (360p) visual media, proving its visual encoder is highly invariant to compression artifacts, yet still fails on the acoustic blindspot.

## Ethical Considerations
*   **Bias:** The underlying FakeAVCeleb dataset is categorized by racial demographic (e.g., African, Asian, Caucasian). Accuracy may vary slightly across these demographics due to representation disparities in the source material.
*   **False Accusations:** Due to the known vulnerabilities to noise and specific manipulation types, utilizing this model in a real-world setting poses a severe risk of falsely accusing authentic media as synthetic, or validating malicious deepfakes as pristine. It must be used purely as a transparent analytical instrument.
