# V16.1 Final Report

## 1. Objective
Complete the V16.1 Audio Robustness workflow, assessing whether adding noise augmentation during fine-tuning could resolve the acoustic instability of the V15.4 baseline.

## 2. V15.4 baseline
The V15.4 model serves as the locked production baseline. While generally performant, it exhibited severe vulnerability to mild ambient noise, prompting this investigation.
- **Checkpoint**: `checkpoints/v14_fullscale/models/best_audio_model.pth`

## 3-6. Candidate Protocols
Four candidates were trained under strictly identical configurations (batch_size=1, same optimizer/loss, same dataset) with varying audio noise augmentation.
- **Candidate A Protocol (Clean)**: Mathematically identical to V14/V15.4 baseline. Retained as reference.
- **Candidate B Protocol (Mild)**: Gaussian noise (std=0.5).
- **Candidate C Protocol (Moderate)**: Gaussian noise (std=1.0).
- **Candidate D Protocol (Mixed)**: Dynamic Gaussian noise with standard deviation uniformly sampled from [0.0, 1.0) per batch.

## 7. Phase 3 Results
(See V16_1_SELECTION_ANALYSIS.md for detailed matrix)
- **Cand A**: Maintained strong clean performance but degraded heavily under noise (FPR spiking to 0.32 at Moderate, 1.0 at Strong).
- **Cand B & D**: Suffered catastrophic failure under noise. While clean AUC remained acceptable, their False Positive Rates skyrocketed to 1.0000 on Moderate/Strong noise. They devolved into predicting 'fake' for nearly all videos.
- **Cand C**: Exhibited a different failure mode. It preserved its False Positive Rate near zero under noise, but its False Negative Rate spiked to 0.9000 (missing 90% of fakes). 

## 8. Robustness Analysis
No candidate achieved balanced robustness. The addition of synthetic Gaussian noise caused the models to either aggressively penalize any noisy real audio (Cand B/D) or passively ignore noisy fake audio (Cand C). 

## 9. Candidate Selection Reasoning
None of the candidates successfully decoupled acoustic noise from deepfake manipulation artifacts. Implementing any of them would severely harm production reliability. Candidate A (V15.4) remains the most balanced overall model.

## 10. Locked-test protocol
Not executed, per strict gating requirements, because no candidate demonstrated meaningful improvement over the V15.4 baseline on the Phase 3 validation set.

## 11. Locked-test result
N/A

## 12. V15.4 vs V16.1 comparison
V15.4 remains vastly superior to all V16.1 candidates in terms of balanced error rate distribution. V16.1 candidates either hallucinated fakes or missed them entirely when subjected to noise.

## 13. Failure cases
The noise augmentation strategy failed. Rather than learning to ignore the noise, the models learned to associate the specific synthetic Gaussian noise distribution directly with the label, causing catastrophic overfitting.

## 14. Limitations
The Gaussian noise synthetic augmentation was too crude. A more sophisticated, physically-modeled ambient noise augmentation (e.g., MUSAN dataset) may be required.

## 15. Scientific validity
The experimental setup was rigorously controlled. All candidates were trained identically except for the augmentation parameter, ensuring the failure mode is cleanly isolated to the augmentation strategy itself.

## 16. Final production recommendation
**Retain V15.4 as the production baseline.** Do not deploy V16.1.

## 17. Exact checkpoint paths
- V15.4 Production: `checkpoints/v14_fullscale/models/best_audio_model.pth`
- V16.1 (Experimental/Abandoned): `checkpoints/v16_1_Cand_*/models/best_audio_model.pth`

## 18. Reproducibility commands
- Train: `python train_v16_1_candidates.py` (Must use batch_size=1)
- Evaluate: `python run_v16_phase3.py`

## 19. Final status
V16.1 is officially classified as EXPERIMENTAL / INVALIDATED. V15.4 remains FROZEN.

---
RESEARCH STATUS: EXPERIMENTAL / INVALIDATED
PRODUCTION BASELINE: V15.4
PRODUCTION CHECKPOINT: checkpoints/v14_fullscale/models/best_audio_model.pth
V16.1 STATUS: ABANDONED
LOCKED TEST STATUS: NOT RUN
BEST VALIDATED ROC-AUC: 0.9145 (V15.4 Clean)
BEST VALIDATED BALANCED ACCURACY: 0.9000 (V15.4 Clean)
FPR: 0.0400 (V15.4 Clean)
FNR: 0.1600 (V15.4 Clean)
KNOWN LIMITATIONS: Extreme vulnerability to ambient audio noise.
REMAINING MATERIAL BLOCKERS: Lack of realistic background noise dataset for training augmentation.
RECOMMENDED NEXT VERSION: V16.2 (incorporating MUSAN environmental noise)
IS THE PROJECT READY TO STOP DEVELOPMENT FOR NOW: YES
