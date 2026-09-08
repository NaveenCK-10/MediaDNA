# Experiment: Phase3_Cand_D_Clean

## Configuration
```json
{
    "exp_name": "Phase3_Cand_D_Clean",
    "checkpoint": "checkpoints/v16_1_Cand_D_Mixed/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.8949
- **PR-AUC (mAP)**: 0.9536
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.6245
- **Fake Mean Score**: 0.9429

### Threshold 0.55
- **ROC-AUC**: 0.8949
- **PR-AUC (mAP)**: 0.9536
- **Accuracy**: 0.8000
- **Balanced Accuracy**: 0.6467
- **F1**: 0.8773
- **Precision**: 0.8125
- **FPR**: 0.6600 (33/50)
- **FNR**: 0.0467 (7/150)
- **Real Mean Score**: 0.6245
- **Fake Mean Score**: 0.9429

### Threshold 0.60
- **ROC-AUC**: 0.8949
- **PR-AUC (mAP)**: 0.9536
- **Accuracy**: 0.8600
- **Balanced Accuracy**: 0.8133
- **F1**: 0.9067
- **Precision**: 0.9067
- **FPR**: 0.2800 (14/50)
- **FNR**: 0.0933 (14/150)
- **Real Mean Score**: 0.6245
- **Fake Mean Score**: 0.9429

### Threshold 0.65
- **ROC-AUC**: 0.8949
- **PR-AUC (mAP)**: 0.9536
- **Accuracy**: 0.8600
- **Balanced Accuracy**: 0.8400
- **F1**: 0.9041
- **Precision**: 0.9296
- **FPR**: 0.2000 (10/50)
- **FNR**: 0.1200 (18/150)
- **Real Mean Score**: 0.6245
- **Fake Mean Score**: 0.9429

