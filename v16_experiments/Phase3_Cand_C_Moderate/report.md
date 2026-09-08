# Experiment: Phase3_Cand_C_Moderate

## Configuration
```json
{
    "exp_name": "Phase3_Cand_C_Moderate",
    "checkpoint": "checkpoints/v16_1_Cand_C_Moderate/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 1.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.8751
- **PR-AUC (mAP)**: 0.9552
- **Accuracy**: 0.8000
- **Balanced Accuracy**: 0.8000
- **F1**: 0.8571
- **Precision**: 0.9231
- **FPR**: 0.2000 (10/50)
- **FNR**: 0.2000 (30/150)
- **Real Mean Score**: 0.4941
- **Fake Mean Score**: 0.5352

### Threshold 0.55
- **ROC-AUC**: 0.8751
- **PR-AUC (mAP)**: 0.9552
- **Accuracy**: 0.4150
- **Balanced Accuracy**: 0.6100
- **F1**: 0.3607
- **Precision**: 1.0000
- **FPR**: 0.0000 (0/50)
- **FNR**: 0.7800 (117/150)
- **Real Mean Score**: 0.4941
- **Fake Mean Score**: 0.5352

### Threshold 0.60
- **ROC-AUC**: 0.8751
- **PR-AUC (mAP)**: 0.9552
- **Accuracy**: 0.3250
- **Balanced Accuracy**: 0.5500
- **F1**: 0.1818
- **Precision**: 1.0000
- **FPR**: 0.0000 (0/50)
- **FNR**: 0.9000 (135/150)
- **Real Mean Score**: 0.4941
- **Fake Mean Score**: 0.5352

### Threshold 0.65
- **ROC-AUC**: 0.8751
- **PR-AUC (mAP)**: 0.9552
- **Accuracy**: 0.2900
- **Balanced Accuracy**: 0.5267
- **F1**: 0.1013
- **Precision**: 1.0000
- **FPR**: 0.0000 (0/50)
- **FNR**: 0.9467 (142/150)
- **Real Mean Score**: 0.4941
- **Fake Mean Score**: 0.5352

