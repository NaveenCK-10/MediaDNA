# Experiment: Phase3_Cand_C_Mild

## Configuration
```json
{
    "exp_name": "Phase3_Cand_C_Mild",
    "checkpoint": "checkpoints/v16_1_Cand_C_Moderate/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.5,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9717
- **Accuracy**: 0.8200
- **Balanced Accuracy**: 0.8733
- **F1**: 0.8647
- **Precision**: 0.9914
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.2333 (35/150)
- **Real Mean Score**: 0.4822
- **Fake Mean Score**: 0.7085

### Threshold 0.55
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9717
- **Accuracy**: 0.7600
- **Balanced Accuracy**: 0.8333
- **F1**: 0.8110
- **Precision**: 0.9904
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.3133 (47/150)
- **Real Mean Score**: 0.4822
- **Fake Mean Score**: 0.7085

### Threshold 0.60
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9717
- **Accuracy**: 0.6900
- **Balanced Accuracy**: 0.7867
- **F1**: 0.7417
- **Precision**: 0.9889
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.4067 (61/150)
- **Real Mean Score**: 0.4822
- **Fake Mean Score**: 0.7085

### Threshold 0.65
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9717
- **Accuracy**: 0.6400
- **Balanced Accuracy**: 0.7533
- **F1**: 0.6870
- **Precision**: 0.9875
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.4733 (71/150)
- **Real Mean Score**: 0.4822
- **Fake Mean Score**: 0.7085

