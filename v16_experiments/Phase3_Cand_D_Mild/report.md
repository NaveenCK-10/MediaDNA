# Experiment: Phase3_Cand_D_Mild

## Configuration
```json
{
    "exp_name": "Phase3_Cand_D_Mild",
    "checkpoint": "checkpoints/v16_1_Cand_D_Mixed/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.5,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9223
- **PR-AUC (mAP)**: 0.9730
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.6987
- **Fake Mean Score**: 0.9565

### Threshold 0.55
- **ROC-AUC**: 0.9223
- **PR-AUC (mAP)**: 0.9730
- **Accuracy**: 0.7600
- **Balanced Accuracy**: 0.5200
- **F1**: 0.8621
- **Precision**: 0.7576
- **FPR**: 0.9600 (48/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.6987
- **Fake Mean Score**: 0.9565

### Threshold 0.60
- **ROC-AUC**: 0.9223
- **PR-AUC (mAP)**: 0.9730
- **Accuracy**: 0.7750
- **Balanced Accuracy**: 0.5833
- **F1**: 0.8657
- **Precision**: 0.7838
- **FPR**: 0.8000 (40/50)
- **FNR**: 0.0333 (5/150)
- **Real Mean Score**: 0.6987
- **Fake Mean Score**: 0.9565

### Threshold 0.65
- **ROC-AUC**: 0.9223
- **PR-AUC (mAP)**: 0.9730
- **Accuracy**: 0.8200
- **Balanced Accuracy**: 0.6933
- **F1**: 0.8875
- **Precision**: 0.8353
- **FPR**: 0.5600 (28/50)
- **FNR**: 0.0533 (8/150)
- **Real Mean Score**: 0.6987
- **Fake Mean Score**: 0.9565

