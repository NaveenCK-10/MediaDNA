# MediaDNA Research Report

## 1. Experimental Setup
- **Test Set**: 303 strictly held-out FakeAVCeleb videos
- **Baseline**: OpenAVFF `best_audio_model.pth`
- **Visual Features**: Laplacian variance (Blur), MAE (Temporal)
- **Fusion Strategy**: 80% OpenAVFF, 20% Visual Anomaly (Experimental Baseline)

## 2. Global Results (Ablation)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| OpenAVFF Baseline | 0.6469 | 0.9524 | 0.6475 | 0.7709 | 0.6944 |
| OpenAVFF + Blur | 0.6238 | 0.9607 | 0.6151 | 0.7500 | 0.7256 |
| OpenAVFF + Temporal | 0.5941 | 0.9641 | 0.5791 | 0.7236 | 0.6945 |
| MediaDNA Fusion | 0.6073 | 0.9649 | 0.5935 | 0.7350 | 0.7134 |

## 3. Category Breakdown: Baseline vs MediaDNA
| Category | Baseline Acc | MediaDNA Acc | Baseline Recall | MediaDNA Recall |
|----------|--------------|--------------|-----------------|-----------------|
| RealVideo-RealAudio | 0.6400 | 0.7600 | 0.0000 | 0.0000 |
| RealVideo-FakeAudio | 0.9643 | 0.9643 | 0.9643 | 0.9643 |
| FakeVideo-RealAudio | 0.2720 | 0.1760 | 0.2720 | 0.1760 |
| FakeVideo-FakeAudio | 0.9520 | 0.9280 | 0.9520 | 0.9280 |

## 4. Conclusion and Blindspot Analysis
The MediaDNA fusion method changed overall accuracy by -4.0%. We observed the following effects on the FakeVideo-RealAudio blindspot: The fusion did not improve the blindspot (change: -9.6%). The experimental 80/20 weighting scheme provides a transparent baseline, but learning an adaptive fusion or decision tree using these visual anomaly signals may be required for robust production use.