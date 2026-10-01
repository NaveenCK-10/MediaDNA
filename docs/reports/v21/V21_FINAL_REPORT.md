# V21 MASTER REPORT: VALIDATION & GENERALIZATION

## 1. Current Baseline
Frozen at OpenAVFF V20.1 structure. Evaluated ACC: 52.5%.

## 2. Reproducibility Status
The harness is deterministic. Fixed decision boundary at 0.60 prevents threshold gaming.

## 3. Dataset Leakage Findings
Train/Val splits overlap across generator families. A disjoint test split is absolutely critical moving forward.

## 4. Error Analysis
FakeVideo-RealAudio is the primary blindspot.

## 5. Modality Ablation
Audio overwhelmingly drives the model. Visual-only collapses to random.

## 6. Audio Specialist Results
No statistically significant uplift observed.

## 7. Fusion Results
Late fusion rejected due to instability.

## 8. Robustness Results
Compression drops performance by 4.5%.

## 9. A/V Desynchronization Results
Model shows little sensitivity to temporal shifts, indicating a failure to genuinely measure cross-modal synchronization.

## 10. Cross-Dataset Results
Performance bounds expected to fall <50% on external wild data.

## 11. Identity/Generator Disjoint Results
Memorization is prevalent.

## 12. Calibration
Currently NOT_VALIDATED. Platt scaling recommended for V22.

## 13. Explainability Validation
Occlusion sensitivity is noisy but functional. Cannot be used for pixel-level bounding.

## 14. Temporal Analysis
Frames are averaged; true temporal recurrent networks are absent.

## 15. Accuracy Improvements
To improve accuracy, we require contrastive pretraining and hard-negative mining on FakeVideo-RealAudio samples.

## 16. Negative Results
Audio specialists and A/V desync analyses failed to yield predictive value.

## 17. Remaining Limitations
Not production ready.

## 18. Exact Commands Executed
`python v21_master_eval.py`

## 19. Exact Files Changed
All V21 output matrices.

## 20. Recommended V22
Implement true cross-attention fusion and Plott scaling calibration.