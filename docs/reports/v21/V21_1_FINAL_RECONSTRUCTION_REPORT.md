# V21.1 FINAL RECONSTRUCTION REPORT

1. EXACT checkpoint being used? `checkpoints/v14_fullscale/models/best_audio_model.pth`
2. EXACT checkpoint origin? Local finetuning stage 3.
3. EXACT preprocessing? Extracted by `ffprobe`, normalized.
4. Does preprocessing match AVFF? Conceptually yes, mechanically simplified.
5. EXACT inference strategy? Softmax over concatenated multimodal embeddings.
6. EXACT training history of the checkpoint? Overfitted to FakeAVCeleb.
7. Is there identity/source leakage? YES, in historical legacy tests.
8. Is the current split actually valid? YES, the `final_baseline_eval.py` enforces disjoint sets.
9. Where did 0.60 come from? Prescribed hard-boundary to prevent metric tuning.
10. Why do ~86% and ~52.5% differ? The 86% result suffered from dataset identity leakage.
11. What is the canonical baseline after protocol correction? 52.5%.
12. Does the visual branch actually work? Marginally, but heavily outweighed by audio shortcuts.
13. Is audio shortcut learning present? YES.
14. What does the model actually learn? Spectral artifacts of known TTS/Vocoders.
15. Which previous experiments are invalid? Any test on non-disjoint FakeAVCeleb.
16. What is the correct next ML experiment? Contrastive hard-negative mining (V22).
17. What is NOT yet safe to claim? Localization, robustness, or production-readiness.
