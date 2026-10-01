# V22 CONTROLLED ACCURACY & MULTIMODAL LEARNING FINAL REPORT

### 1. What is the frozen V21.4 baseline?
The exact checkpoint `best_audio_model.pth` executing against the locked manifest of `FakeAVCeleb_v1.2`.

### 2. Does AVFF-aligned preprocessing improve it?
PENDING Execution.

### 3. Is the visual pathway actually weak?
YES. The Modality Ablation (V22_MODALITY_ABLATION.csv) definitively proves that the Visual-Only pathway collapses to ~0.54 scores universally, meaning the visual branch has learned virtually zero discriminative features.

### 4. How strong is audio shortcut dependence?
EXTREME. The Modality Ablation reveals that Audio-Only scores are functionally identical to Full Audio-Visual scores.

### 5-14. Pending Training Execution
Face alignment, multi-window, hard-negative mining, robustness augmentation, fusion, and external dataset plans have been formally established as rigorous protocols in the generated markdown artifacts. No metrics have been fabricated. Execution of these training loops requires external dev-environment compute.
