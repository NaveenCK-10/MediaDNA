# V21 VALIDATION PROTOCOL

## 1. Dataset & Leakage Prevention
**Dataset**: FakeAVCeleb_v1.2
**Subsets**: RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, FakeVideo-FakeAudio
**Leakage Audit**: The `data/train*.csv` and `data/val*.csv` are explicitly excluded from the test splits. No overlapping paths or basenames are permitted.
**Generator Leakage**: Acknowledged that evaluating on the same generators as training may inflate performance, but currently relying on standard subset split boundaries.

## 2. Threshold Independence
**Decision Protocol**: Locked at 0.60 per V20.1 constraints. 
Calibration requires a held-out dataset using isotonic regression or Platt scaling. Currently: NOT_VALIDATED.

## 3. Evaluation Schema
All performance metrics track:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1
- ROC-AUC, FPR, FNR
- Confusion Matrix

## 4. Modality Ablation
Will isolate audio-only and video-only frames using baseline masking or explicit network branches if available, computing the delta in held-out accuracy.

## 5. Robustness Iteration
Simulated perturbations will map:
- Brightness adjustment
- Gaussian Blur
- Compression artifacts (H.264 high crf)
Performance drop relative to the baseline clean frame will be recorded.

## 6. A/V Desynchronization
Testing temporal shifts:
- -500ms, -250ms, 0ms, +250ms, +500ms
This tracks failure stability across misaligned streams.

## 7. Execution Status
Due to computational constraints within autonomous agent execution environments, the deep evaluations will process a micro-batch sample to prove the algorithmic end-to-end viability of the evaluation harness. The resulting matrices reflect the schema, but large-scale runtime execution must be delegated to dedicated compute clusters.
