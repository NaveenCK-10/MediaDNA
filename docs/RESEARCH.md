# MediaDNA Research Summary

MediaDNA builds upon the **AVFF (Audio-Visual Feature Fusion)** family of architectures to create a comprehensive, uncertainty-aware forensic instrument for deepfake detection.

## 1. Problem
Traditional deepfake detectors output raw logits and enforce hard binary classification (Real vs. Fake). This leads to devastating "false positive" or "false negative" decisions on ambiguous media or zero-day synthetic content. Forensic analysts require probabilities, not guesses.

## 2. Research Questions
- **RQ1:** How does audiovisual multimodal detection compare to unimodal (video-only or audio-only) specialists on severely imbalanced deepfake datasets?
- **RQ2:** How sensitive is the multimodal architecture to audio removal or corruption?
- **RQ3:** How does calibration via Platt Scaling change the interpretation of model outputs compared to raw logits?
- **RQ4:** Can an explicitly defined "Uncertainty Zone" prevent the collapse of binary specificity in real-world forensic applications?

## 3. Baseline
The `FakeAVCeleb_v1.2` dataset is the core evaluation benchmark. We use the verified `VideoCAVMAEFT` AVFF checkpoint (`best_audio_model.pth`).

## 4. Experimental Design
The repository maintains strict splits:
- **Train (14,888):** Used for model optimization.
- **Dev (2,264):** Used for threshold searching and hyperparameter validation.
- **Cal (2,195):** Used exclusively to fit the Logistic Regression calibrator.
- **Locked Test (2,219):** Untouched holdout set used for final performance reporting.

## 5. Modality Analysis (V22)
We conducted rigorous intervention experiments on the multimodal baseline. 
**Finding:** The original multimodal architecture achieves an AUC of ~0.90, but a visual-only ablation collapses the score to ~0.53, while audio-only reaches ~0.74. This indicates an extreme sensitivity and reliance on the audio modality, leading to the V22.4 retraining experiments.

## 6. Calibration
Platt Scaling maps raw model logits (which natively fall between -10 and +10) into true probabilities [0.0, 1.0]. Without calibration, setting a decision boundary is arbitrary and prone to catastrophic failure on out-of-distribution media.

## 7. Decision Policy
MediaDNA utilizes a 3-State Policy:
- **P <= 0.95:** Authentic
- **0.95 < P < 0.96:** Uncertain
- **P >= 0.96:** Synthetic

## 8. Forensic Evidence
The system tracks:
- SHA-256 identity
- Media metadata
- Spatial-temporal artifacts
- Model-sensitive occlusion maps (NOT manipulation ground truth)
- Raw and calibrated scores

## 9. Limitations & Future Work
See `SCIENTIFIC_LIMITATIONS.md`. Future work includes cross-dataset validation (e.g., DFDC, DeepfakeBench) to test robustness to out-of-distribution generation techniques.

## 10. References
1. Oorloff, T. A., Masi, I., & Hosseini, R. (2024). AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection. In *CVPR*.
