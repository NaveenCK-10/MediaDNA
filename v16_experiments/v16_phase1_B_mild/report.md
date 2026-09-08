# Experiment: v16_phase1_B_mild

## Configuration
```json
{
    "exp_name": "v16_phase1_B_mild",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.5,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9095
- **PR-AUC (mAP)**: 0.9673
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5635
- **Fake Mean Score**: 0.8652

### Threshold 0.55
- **ROC-AUC**: 0.9095
- **PR-AUC (mAP)**: 0.9673
- **Accuracy**: 0.7900
- **Balanced Accuracy**: 0.6400
- **F1**: 0.8704
- **Precision**: 0.8103
- **FPR**: 0.6600 (33/50)
- **FNR**: 0.0600 (9/150)
- **Real Mean Score**: 0.5635
- **Fake Mean Score**: 0.8652

### Threshold 0.60
- **ROC-AUC**: 0.9095
- **PR-AUC (mAP)**: 0.9673
- **Accuracy**: 0.8650
- **Balanced Accuracy**: 0.9033
- **F1**: 0.9018
- **Precision**: 0.9920
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.1733 (26/150)
- **Real Mean Score**: 0.5635
- **Fake Mean Score**: 0.8652

### Threshold 0.65
- **ROC-AUC**: 0.9095
- **PR-AUC (mAP)**: 0.9673
- **Accuracy**: 0.8350
- **Balanced Accuracy**: 0.8833
- **F1**: 0.8773
- **Precision**: 0.9916
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.2133 (32/150)
- **Real Mean Score**: 0.5635
- **Fake Mean Score**: 0.8652

