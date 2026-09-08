# Experiment: Phase3_Cand_C_Strong

## Configuration
```json
{
    "exp_name": "Phase3_Cand_C_Strong",
    "checkpoint": "checkpoints/v16_1_Cand_C_Moderate/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 2.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.7679
- **PR-AUC (mAP)**: 0.9025
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5474
- **Fake Mean Score**: 0.5825

### Threshold 0.55
- **ROC-AUC**: 0.7679
- **PR-AUC (mAP)**: 0.9025
- **Accuracy**: 0.7400
- **Balanced Accuracy**: 0.7000
- **F1**: 0.8182
- **Precision**: 0.8603
- **FPR**: 0.3800 (19/50)
- **FNR**: 0.2200 (33/150)
- **Real Mean Score**: 0.5474
- **Fake Mean Score**: 0.5825

### Threshold 0.60
- **ROC-AUC**: 0.7679
- **PR-AUC (mAP)**: 0.9025
- **Accuracy**: 0.4300
- **Balanced Accuracy**: 0.6000
- **F1**: 0.4062
- **Precision**: 0.9286
- **FPR**: 0.0600 (3/50)
- **FNR**: 0.7400 (111/150)
- **Real Mean Score**: 0.5474
- **Fake Mean Score**: 0.5825

### Threshold 0.65
- **ROC-AUC**: 0.7679
- **PR-AUC (mAP)**: 0.9025
- **Accuracy**: 0.3200
- **Balanced Accuracy**: 0.5467
- **F1**: 0.1707
- **Precision**: 1.0000
- **FPR**: 0.0000 (0/50)
- **FNR**: 0.9067 (136/150)
- **Real Mean Score**: 0.5474
- **Fake Mean Score**: 0.5825

