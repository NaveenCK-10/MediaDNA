# Experiment: v16_phase1_C_moderate

## Configuration
```json
{
    "exp_name": "v16_phase1_C_moderate",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 1.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.8877
- **PR-AUC (mAP)**: 0.9631
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5952
- **Fake Mean Score**: 0.6606

### Threshold 0.55
- **ROC-AUC**: 0.8877
- **PR-AUC (mAP)**: 0.9631
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5952
- **Fake Mean Score**: 0.6606

### Threshold 0.60
- **ROC-AUC**: 0.8877
- **PR-AUC (mAP)**: 0.9631
- **Accuracy**: 0.8350
- **Balanced Accuracy**: 0.7833
- **F1**: 0.8896
- **Precision**: 0.8926
- **FPR**: 0.3200 (16/50)
- **FNR**: 0.1133 (17/150)
- **Real Mean Score**: 0.5952
- **Fake Mean Score**: 0.6606

### Threshold 0.65
- **ROC-AUC**: 0.8877
- **PR-AUC (mAP)**: 0.9631
- **Accuracy**: 0.6050
- **Balanced Accuracy**: 0.7367
- **F1**: 0.6425
- **Precision**: 1.0000
- **FPR**: 0.0000 (0/50)
- **FNR**: 0.5267 (79/150)
- **Real Mean Score**: 0.5952
- **Fake Mean Score**: 0.6606

