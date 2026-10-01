# MediaDNA V22.3 Collapse: Root Cause Analysis

## Executive Summary
The V22.3 MediaDNA detector collapsed into a state of near-random predictions (Balanced Accuracy ≈ 0.50, MCC ≈ 0.0) due to a combination of severe class imbalance, extreme undertraining, missing loss weighting, and architectural incompatibility with the loaded V14 weights. Furthermore, the decision to separate the detector into independent visual and audio specialists discarded the cross-modal fusion capabilities that were the primary source of discrimination in the original OpenAVFF baseline.

## Critical Findings

### 1. Severe Class Imbalance with Unweighted Loss
- **Imbalance**: The `v22_2_train.csv` dataset contains 14,538 fake samples and only 350 real samples (a 41.5:1 ratio, or 97.6% fake).
- **Missing Correction**: The models were trained using a standard `BCEWithLogitsLoss()` without any `pos_weight` parameter or class balancing techniques.
- **Consequence**: The network trivially learned the dataset prior. By uniformly predicting "fake" (probability ≈ 0.78), it achieved a superficially high training accuracy and F1 score, masking the fact that it misclassified 100% of the real samples (RVRA accuracy = 0.0).

### 2. Extreme Undertraining (25-Batch Limit)
- **Hard Coded Limit**: The training scripts (`tools/v22_3B_visual_train.py` and `tools/v22_3C_audio_train.py`) contained a hard-coded break statement: `if i > 25: break`.
- **Sample Count**: With a batch size of 4, the models were only seeing 100 samples per epoch out of the available 14,888.
- **Epoch Count**: The training was limited to exactly 1 epoch.
- **Consequence**: The models underwent almost zero meaningful optimization. The V22.3 weights are effectively still at their random initialization state (with a slight bias towards the "fake" class).

### 3. Checkpoint Loading Incompatibility
- **Architecture Mismatch**: The training scripts loaded the V14 `best_audio_model.pth` checkpoint into the `VisualSpecialist` and `AudioSpecialist` architectures.
- **Lost Weights**: Because the specialists lacked the cross-modal fusion blocks (`a2v`, `v2a`) and had renamed components compared to the V14 `VideoCAVMAEFT` model, a large portion of the learned weights failed to load.
- **Strict=False Fallacy**: The use of `strict=False` suppressed the errors, silently allowing the models to run with randomly initialized layers in critical pathways.

### 4. Intentional Removal of Cross-Modal Discrimination
- **Specialist Design Flaw**: The decision to split the pipeline into independent visual and audio specialists fundamentally crippled its performance.
- **Baseline Strength**: The original AVFF model achieved a threshold-free ROC-AUC of 0.897 on the DEV set. A significant portion of this performance comes from detecting audio-visual mismatch (e.g., RealVideo-FakeAudio, which achieves 0.98 accuracy).
- **Consequence**: The unimodal specialists cannot, by definition, detect cross-modal inconsistencies, capping their theoretical maximum performance well below the AVFF baseline.

## Resolution
The recovery strategy explicitly rejects the V22.3 specialist architecture in favor of restoring the joint AVFF model (`VideoCAVMAEFT`) as the primary detection engine, while preserving the specialist logic purely for diagnostic evidence reporting. The pipeline has been updated to include proper probability calibration (Platt scaling) and a robust three-tier decision threshold policy.
