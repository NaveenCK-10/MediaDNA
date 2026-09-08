# Experiment: exp_003_harness_verification

## Configuration
```json
{
    "exp_name": "exp_003_harness_verification",
    "checkpoint": "checkpoints/v14_fullscale/models/best_audio_model.pth",
    "csv": "data/test_locked_v14.csv",
    "mode": "normal",
    "noise_level": 0.0,
    "batch_size": 16
}
```

## Metrics
### Threshold 0.50
- **ROC-AUC**: 0.9261
- **PR-AUC (mAP)**: 0.9783
- **Accuracy**: 0.7500
- **Balanced Accuracy**: 0.5000
- **F1**: 0.8571
- **Precision**: 0.7500
- **FPR**: 1.0000 (100/100)
- **FNR**: 0.0000 (0/300)
- **Real Mean Score**: 0.5415
- **Fake Mean Score**: 0.9160

### Threshold 0.55
- **ROC-AUC**: 0.9261
- **PR-AUC (mAP)**: 0.9783
- **Accuracy**: 0.8600
- **Balanced Accuracy**: 0.8533
- **F1**: 0.9028
- **Precision**: 0.9420
- **FPR**: 0.1600 (16/100)
- **FNR**: 0.1333 (40/300)
- **Real Mean Score**: 0.5415
- **Fake Mean Score**: 0.9160

### Threshold 0.60
- **ROC-AUC**: 0.9261
- **PR-AUC (mAP)**: 0.9783
- **Accuracy**: 0.8625
- **Balanced Accuracy**: 0.8883
- **F1**: 0.9013
- **Precision**: 0.9767
- **FPR**: 0.0600 (6/100)
- **FNR**: 0.1633 (49/300)
- **Real Mean Score**: 0.5415
- **Fake Mean Score**: 0.9160

### Threshold 0.65
- **ROC-AUC**: 0.9261
- **PR-AUC (mAP)**: 0.9783
- **Accuracy**: 0.8600
- **Balanced Accuracy**: 0.8933
- **F1**: 0.8986
- **Precision**: 0.9841
- **FPR**: 0.0400 (4/100)
- **FNR**: 0.1733 (52/300)
- **Real Mean Score**: 0.5415
- **Fake Mean Score**: 0.9160

