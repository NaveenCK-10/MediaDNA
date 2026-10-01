# MEDIADNA_EXACT_VIDEO_DIAGNOSTIC

## 1. Exact Video Details
**File:** `Create_a_photorealistic_AI_gen.mp4`
**Path:** `C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4`
**SHA-256:** `[Pending Test Output]`

## 2. Current Pipeline Architecture
1. **Video**
2. **V22.3 Visual Specialist**
3. **V22.3 Audio Specialist**
4. **0.5 visual + 0.5 audio late fusion**
5. **V22.3 calibration (Repaired LogisticRegression Logit Calibrator)**
6. **3-state decision policy**

## 3. Findings: Calibration Collapse & Model Limitation
The previous V22_3_PLATT_CALIBRATOR.pkl was mathematically collapsed, forcing almost all inputs to evaluate to exactly ~0.97728. This resulted in false SYNTHETIC flags for known REAL samples when using the `P >= 0.99` threshold (or UNCERTAIN due to the upper bound).

We rebuilt the calibration using a statistically sound `LogisticRegression` fitted on logit-transformed raw fusion scores with `class_weight='balanced'`. 

The repaired calibrator revealed a fundamental model limitation: the V22.3 pipeline raw scores have almost zero separation power on the calibration dataset.
- Real Samples (Mean Calibrated Prob): 0.4977
- Synthetic Samples (Mean Calibrated Prob): 0.5022

As a result, the mathematically sound output for the exact video (and most videos) is **UNCERTAIN**, because the underlying specialist models do not provide enough distinguishing evidence.

## 4. Pipeline Repair
- **PDF Report Engine**: Completely rebuilt. Decoupled from LLM dependence. Uses deterministic structured JSON and Playwright HTML-to-PDF rendering with a premium dark cyber-forensic aesthetic. 
- **NVIDIA Integration**: Decoupled to an optional `NvidiaNarrativeGenerator` module. If the API fails or times out, a robust deterministic fallback narrative is used.
- **Label Mapping**: Repaired decision thresholds to map correctly to the new calibrator (Authentic < 0.3, Uncertain 0.3-0.7, Synthetic > 0.7).
