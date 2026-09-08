# MediaDNA Research Report: OpenAVFF Robustness and Fusion Efficacy

## 1. Abstract
This report documents the performance of the OpenAVFF (VideoCAVMAEFT) multimodal deepfake detection model on the FakeAVCeleb dataset, augmented by the experimental MediaDNA visual heuristic fusion pipeline. We found that while the OpenAVFF core provides a strong baseline, its vulnerability to unmanipulated audio in video deepfakes limits its real-world robustness. The MediaDNA experimental fusion attempted to penalize these anomalies but ultimately degraded the overall Area Under Curve (AUC) from the baseline.

## 2. Problem Statement
Multimodal deepfakes manipulate spatial geometry (video) and acoustic signatures (audio). Modern detection architectures often struggle when one modality is left untouched, specifically when realistic audio is dubbed over a synthetically generated video track. The goal of this research was to benchmark the OpenAVFF architecture against these precise permutations and attempt to correct its blindspots using classical computer vision fusion.

## 3. Dataset & Preprocessing
The model was trained and evaluated strictly on `FakeAVCeleb_v1.2`, which isolates modalities into four subcategories:
1.  **RealVideo-RealAudio (Real)**
2.  **FakeVideo-FakeAudio (Fake)**
3.  **FakeVideo-RealAudio (Fake)**
4.  **RealVideo-FakeAudio (Fake)**

Video frames were sampled at 16 equidistant intervals and center-cropped to 224x224. Audio was extracted at 16kHz and converted into 80-dimensional Kaldi log-mel filterbanks, interpolated to 1024 temporal frames.

## 4. Architecture
The underlying core is a **VideoCAVMAEFT**, which utilizes:
*   A spatial-temporal visual encoder.
*   An acoustic mel-spectrogram encoder.
*   Cross-attention heads (Audio-to-Video and Video-to-Audio).
*   An MLP classification head yielding a binary logit.

The experimental **MediaDNA Fusion Layer** appended an analytical algorithm calculating frame-to-frame Mean Absolute Error (MAE) and spatial Laplacian variance to mathematically assess visual blur or temporal glitching independently of the neural network.

## 5. Evaluation Results

Testing was conducted on a held-out test split comprising 5,145 samples.

### Baseline OpenAVFF Performance
*   **Global Accuracy:** 64.69%
*   **Global ROC-AUC:** 0.6944
*   **RealVideo-RealAudio Accuracy:** 64.0%
*   **RealVideo-FakeAudio Accuracy:** 96.43%
*   **FakeVideo-FakeAudio Accuracy:** 95.20%
*   **FakeVideo-RealAudio Accuracy:** 27.20%

### MediaDNA Experimental Fusion Performance
*   **Global Accuracy:** 60.73%
*   **RealVideo-RealAudio Accuracy:** 43.19%
*   **FakeVideo-FakeAudio Accuracy:** 99.40%
*   **FakeVideo-RealAudio Accuracy:** 56.62%

## 6. Failure Analysis & Ablation
The evaluation identified a severe structural blindspot: **FakeVideo-RealAudio**. 
The baseline model achieves only 27.20% accuracy on videos where the visual track is manipulated but the audio track remains authentic. The model's cross-attention mechanism appears to heavily over-index on acoustic authenticity; if the audio sounds real and matches the phoneme rhythm, the model suppresses visual anomalies and declares the video "REAL".

The experimental MediaDNA fusion attempted to correct this by applying a 20% weight to strict visual anomalies (like blur variance). While this successfully improved FakeVideo-RealAudio accuracy from 27.20% to 56.62%, it catastrophically degraded RealVideo-RealAudio accuracy from 64.0% to 43.19%, resulting in a net-negative ROC-AUC score.

## 7. Robustness Red-Team Evaluation
Beyond the test set, the model was stress-tested against deliberate corruptions:
*   **Audio Noise (SNR 10dB):** Caused total systemic collapse, pushing predictions heavily toward "Fake" regardless of ground truth.
*   **Video Compression (CRF 40):** The model remained surprisingly resilient to heavy H.264 artifacting, maintaining a confident prediction on genuine deepfakes.

## 8. Limitations & Future Work
The current OpenAVFF architecture is **not production-ready** for adversarial environments. It acts more as an acoustic validator than a true multimodal deepfake detector.

Future iterations must explore:
1.  **Modality Dropout during Training:** Forcing the network to rely on the visual encoder by randomly zeroing out the acoustic embeddings during the training phase.
2.  **Hard Negative Mining:** Over-sampling the FakeVideo-RealAudio quadrant to penalize the network for ignoring visual artifacts.
3.  **Audio Noise Augmentation:** The model must be trained with synthetic Gaussian and environmental noise to prevent complete failure in low-SNR environments.
