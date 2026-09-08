# Experiment: exp_002_audio_noise_test

## Configuration
```json
{
    "exp_name": "exp_002_audio_noise_test",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.5,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9704
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5630
- **Fake Mean Score**: 0.8652

### Threshold 0.55
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9704
- **Accuracy**: 0.8000
- **Balanced Accuracy**: 0.6533
- **F1**: 0.8765
- **Precision**: 0.8161
- **FPR**: 0.6400 (32/50)
- **FNR**: 0.0533 (8/150)
- **Real Mean Score**: 0.5630
- **Fake Mean Score**: 0.8652

### Threshold 0.60
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9704
- **Accuracy**: 0.8650
- **Balanced Accuracy**: 0.9033
- **F1**: 0.9018
- **Precision**: 0.9920
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.1733 (26/150)
- **Real Mean Score**: 0.5630
- **Fake Mean Score**: 0.8652

### Threshold 0.65
- **ROC-AUC**: 0.9161
- **PR-AUC (mAP)**: 0.9704
- **Accuracy**: 0.8250
- **Balanced Accuracy**: 0.8767
- **F1**: 0.8689
- **Precision**: 0.9915
- **FPR**: 0.0200 (1/50)
- **FNR**: 0.2267 (34/150)
- **Real Mean Score**: 0.5630
- **Fake Mean Score**: 0.8652

