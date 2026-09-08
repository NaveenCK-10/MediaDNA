# Experiment: Phase3_Cand_A_Moderate

## Configuration
```json
{
    "exp_name": "Phase3_Cand_A_Moderate",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 1.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.8910
- **PR-AUC (mAP)**: 0.9643
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5947
- **Fake Mean Score**: 0.6611

### Threshold 0.55
- **ROC-AUC**: 0.8910
- **PR-AUC (mAP)**: 0.9643
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5947
- **Fake Mean Score**: 0.6611

### Threshold 0.60
- **ROC-AUC**: 0.8910
- **PR-AUC (mAP)**: 0.9643
- **Accuracy**: 0.8300
- **Balanced Accuracy**: 0.7800
- **F1**: 0.8859
- **Precision**: 0.8919
- **FPR**: 0.3200 (16/50)
- **FNR**: 0.1200 (18/150)
- **Real Mean Score**: 0.5947
- **Fake Mean Score**: 0.6611

### Threshold 0.65
- **ROC-AUC**: 0.8910
- **PR-AUC (mAP)**: 0.9643
- **Accuracy**: 0.5900
- **Balanced Accuracy**: 0.7267
- **F1**: 0.6239
- **Precision**: 1.0000
- **FPR**: 0.0000 (0/50)
- **FNR**: 0.5467 (82/150)
- **Real Mean Score**: 0.5947
- **Fake Mean Score**: 0.6611

