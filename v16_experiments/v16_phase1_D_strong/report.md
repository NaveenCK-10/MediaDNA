# Experiment: v16_phase1_D_strong

## Configuration
```json
{
    "exp_name": "v16_phase1_D_strong",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 2.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.7555
- **PR-AUC (mAP)**: 0.9010
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.6650
- **Fake Mean Score**: 0.6982

### Threshold 0.55
- **ROC-AUC**: 0.7555
- **PR-AUC (mAP)**: 0.9010
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.6650
- **Fake Mean Score**: 0.6982

### Threshold 0.60
- **ROC-AUC**: 0.7555
- **PR-AUC (mAP)**: 0.9010
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.6650
- **Fake Mean Score**: 0.6982

### Threshold 0.65
- **ROC-AUC**: 0.7555
- **PR-AUC (mAP)**: 0.9010
- **Accuracy**: 0.7450
- **Balanced Accuracy**: 0.5767
- **F1**: 0.8431
- **Precision**: 0.7829
- **FPR**: 0.7600 (38/50)
- **FNR**: 0.0867 (13/150)
- **Real Mean Score**: 0.6650
- **Fake Mean Score**: 0.6982

