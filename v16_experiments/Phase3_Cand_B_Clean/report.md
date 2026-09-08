# Experiment: Phase3_Cand_B_Clean

## Configuration
```json
{
    "exp_name": "Phase3_Cand_B_Clean",
    "checkpoint": "checkpoints/v16_1_Cand_B_Mild/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9216
- **PR-AUC (mAP)**: 0.9726
- **Accuracy**: 0.7650
- **Balanced Accuracy**: 0.5433
- **F1**: 0.8630
- **Precision**: 0.7668
- **FPR**: 0.9000 (45/50)
- **FNR**: 0.0133 (2/150)
- **Real Mean Score**: 0.5806
- **Fake Mean Score**: 0.9316

### Threshold 0.55
- **ROC-AUC**: 0.9216
- **PR-AUC (mAP)**: 0.9726
- **Accuracy**: 0.8350
- **Balanced Accuracy**: 0.7167
- **F1**: 0.8966
- **Precision**: 0.8462
- **FPR**: 0.5200 (26/50)
- **FNR**: 0.0467 (7/150)
- **Real Mean Score**: 0.5806
- **Fake Mean Score**: 0.9316

### Threshold 0.60
- **ROC-AUC**: 0.9216
- **PR-AUC (mAP)**: 0.9726
- **Accuracy**: 0.8650
- **Balanced Accuracy**: 0.8167
- **F1**: 0.9103
- **Precision**: 0.9073
- **FPR**: 0.2800 (14/50)
- **FNR**: 0.0867 (13/150)
- **Real Mean Score**: 0.5806
- **Fake Mean Score**: 0.9316

### Threshold 0.65
- **ROC-AUC**: 0.9216
- **PR-AUC (mAP)**: 0.9726
- **Accuracy**: 0.8550
- **Balanced Accuracy**: 0.8500
- **F1**: 0.8990
- **Precision**: 0.9416
- **FPR**: 0.1600 (8/50)
- **FNR**: 0.1400 (21/150)
- **Real Mean Score**: 0.5806
- **Fake Mean Score**: 0.9316

