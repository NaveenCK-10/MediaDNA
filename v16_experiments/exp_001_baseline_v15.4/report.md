# Experiment: exp_001_baseline_v15.4

## Configuration
```json
{
    "exp_name": "exp_001_baseline_v15.4",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/val_v14.csv",
    "mode": "normal",
    "noise_level": 0.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9145
- **PR-AUC (mAP)**: 0.9702
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (50/50)
- **FNR**: 0.0000 (0/150)
- **Real Mean Score**: 0.5449
- **Fake Mean Score**: 0.9199

### Threshold 0.55
- **ROC-AUC**: 0.9145
- **PR-AUC (mAP)**: 0.9702
- **Accuracy**: 0.8500
- **Balanced Accuracy**: 0.8533
- **F1**: 0.8944
- **Precision**: 0.9478
- **FPR**: 0.1400 (7/50)
- **FNR**: 0.1533 (23/150)
- **Real Mean Score**: 0.5449
- **Fake Mean Score**: 0.9199

### Threshold 0.60
- **ROC-AUC**: 0.9145
- **PR-AUC (mAP)**: 0.9702
- **Accuracy**: 0.8700
- **Balanced Accuracy**: 0.9000
- **F1**: 0.9065
- **Precision**: 0.9844
- **FPR**: 0.0400 (2/50)
- **FNR**: 0.1600 (24/150)
- **Real Mean Score**: 0.5449
- **Fake Mean Score**: 0.9199

### Threshold 0.65
- **ROC-AUC**: 0.9145
- **PR-AUC (mAP)**: 0.9702
- **Accuracy**: 0.8650
- **Balanced Accuracy**: 0.8967
- **F1**: 0.9025
- **Precision**: 0.9843
- **FPR**: 0.0400 (2/50)
- **FNR**: 0.1667 (25/150)
- **Real Mean Score**: 0.5449
- **Fake Mean Score**: 0.9199

