# V22.3 Pre-Repair Sanity Check

## 1. Overview
This document records the sanity check results for the original (broken) implementations of the `VisualSpecialist` and `AudioSpecialist` in the V22.3 architecture.

## 2. Experimental Setup
- **Tool**: `tools/run_fast_sanity.py`
- **Method**: Train the broken variants (`BrokenVisualSpecialist` and `BrokenAudioSpecialist`) which pool over `dim=-1` instead of `dim=1`.
- **Data**: 4 random tensors simulating the expected input shapes, with balanced binary labels.
- **Epochs**: 100

## 3. Results

### Visual Specialist (Pre-Repair)
- **Initial Loss**: 0.693
- **Final Loss**: 0.719
- **Gradient Norm**: 0.331
- **Parameter Delta**: 19.92
- **Training Accuracy (Batch)**: 0.25

### Audio Specialist (Pre-Repair)
- **Initial Loss**: 0.691
- **Final Loss**: 0.731
- **Gradient Norm**: 0.351
- **Parameter Delta**: 20.67
- **Training Accuracy (Batch)**: 0.25

## 4. Observations
The broken pooling across the channel dimension (`dim=-1`) collapses the rich 768-dimensional features, leaving the models unable to correctly minimize the cross-entropy loss over 100 epochs even on a trivial 4-sample mock dataset. This confirms that the V22.3 models suffered from a fundamental tensor dimension bug prior to repair.
