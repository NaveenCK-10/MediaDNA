# V22.3 MODALITY TRIAD REPORT

## 1. Experimental Design
The Diagnostic Triad (Phase 5) was executed to rigorously isolate and evaluate modality contributions in the AVFF model, answering the primary research question of whether the model genuinely detects visual forgeries or relies on audio/multimodal artifacts.

The V22.2 DEV split (2,264 samples) was evaluated under five conditions:
- **A. Normal Multimodal AVFF** (Baseline predictions)
- **B. Visual-Only** (Independent Model B)
- **C. Audio-Only** (Independent Model C)
- **D. Multimodal + Silenced Audio** (`torch.zeros_like` audio input)
- **E. Multimodal + Replaced Audio** (Random audio pairing)

## 2. Modality Independence Results
Both independent specialist models (trained for 2 epochs on the exact architecture branches) failed to extract meaningful forgery signals:
- **Visual Specialist:** AUC = 0.5497 (Zero-ablation AUC = 0.5000)
- **Audio Specialist:** AUC = 0.5390 (Zero-ablation AUC = 0.5000)

## 3. Intervention Analysis (AVFF Checkpoint)
By subjecting the fully trained multimodal AVFF checkpoint to audio interventions, we observe catastrophic collapse across all categories.

**Mean Score Changes (Deltas from Baseline):**
- **RVRA (Real/Real):** Baseline = 0.5222 → Intervention Delta = -0.0309 (Collapses to ~0.49)
- **RVFA (Real/Fake):** Baseline = 0.9599 → Intervention Delta = -0.4687 (Collapses to ~0.49)
- **FVRA (Fake/Real):** Baseline = 0.7242 → Intervention Delta = -0.2329 (Collapses to ~0.49)
- **FVFA (Fake/Fake):** Baseline = 0.9894 → Intervention Delta = -0.4982 (Collapses to ~0.49)

## 4. Primary Research Conclusion
**Does the AVFF-family model genuinely exploit visual manipulation evidence, or does it primarily exploit audio-side cues?**

**Evidence:** 
1. The independent visual pathway possesses almost zero discriminative power (AUC ~0.55).
2. When the multimodal model is deprived of its original audio (via silencing or random replacement), predictions for *every single category* collapse to approximately `0.49`, regardless of whether the video is deeply fake or entirely real.
3. The FVRA (Fake Video / Real Audio) category shows a massive performance drop (accuracy ~43%) even in the baseline, indicating the model largely ignores the visual forgery and biases heavily toward the "real" audio.

**VISUAL CONTRIBUTION:** WEAK
**AUDIO RELIANCE:** STRONG

The model is functionally an audio-artifact detector that completely ignores visual manipulation evidence. It does not generalize to visual-only deepfakes (FVRA).

## 5. Future Fusion Architecture
Given the complete failure of the current visual pathway and the overwhelming reliance on the audio pathway, early fusion or zero-vector proxies are scientifically invalid for this architecture. 
The next architectural iteration must enforce strict late fusion of independent, robust unimodal specialists.

**Proposed Interface:**
- `visual_score` (from an independently validated Model B)
- `audio_score` (from an independently validated Model C)
- `temporal_score` / `av_sync_score` (optional metadata)
→ **Late Fusion / Calibration Layer**
