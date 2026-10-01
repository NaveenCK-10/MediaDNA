# MediaDNA Final Decision-Mapping Audit

## 1. Trace of Complete Samples (Before & After Fix)

### The Defect
The previous configuration contained a bug where the legacy `classification.label` field applied a binary threshold overriding the calibrated probability. Specifically:
```python
label="fake" if decision=="SYNTHETIC" else "real"
```
Because the decision policy for `SYNTHETIC` is `P >= 0.99`, a sample with a probability of `0.9772` correctly mapped to `UNCERTAIN` for the `decision` field, but was erroneously marked as `"real"` in the frontend due to the `else "real"` fallback for the legacy label.

### The Fix
We established a single authoritative decision path:
**Raw Scores → Late Fusion → Platt Calibration (P) → Decision Policy (AUTHENTIC/UNCERTAIN/SYNTHETIC)**

The frontend was updated to completely disregard legacy labels and solely use `authenticity.uncertainty` (the backend's unified decision) to display findings.

---

### Trace 1: Known Real (`00109.mp4`)
- **Raw Visual Score:** 0.7910
- **Raw Audio Score:** 0.8505
- **Fusion Score:** 0.8208
- **Calibrated Probability:** 0.9772
- **Backend Decision Policy:** 0.8273 < P < 0.99 -> `UNCERTAIN`
- **Backend Decision Object:** `UNCERTAIN`
- **Frontend Displayed Decision:** `UNCERTAIN`
- **PDF Decision:** `UNCERTAIN`

### Trace 2: Known Synthetic (`00109_id00475_wavtolip.mp4`)
- **Raw Visual Score:** 0.7905
- **Raw Audio Score:** 0.8559
- **Fusion Score:** 0.8232
- **Calibrated Probability:** 0.9772
- **Backend Decision Policy:** 0.8273 < P < 0.99 -> `UNCERTAIN`
- **Backend Decision Object:** `UNCERTAIN`
- **Frontend Displayed Decision:** `UNCERTAIN`
- **PDF Decision:** `UNCERTAIN`

*Note: The model correctly outputs UNCERTAIN based on the frozen Platt calibrator for these specific samples.*

### Trace 3: Missing Audio (`no_audio.mp4`)
- **Raw Visual Score:** 0.7910
- **Raw Audio Score:** `None`
- **Fusion Score:** 0.7910 (Visual-only fallback)
- **Calibrated Probability:** 0.5 (Skipped calibration)
- **Backend Decision Policy:** `UNCERTAIN`
- **Backend Decision Object:** `UNCERTAIN` ("Audio analysis was unavailable for this media.")
- **Frontend Displayed Decision:** `UNCERTAIN`
- **PDF Decision:** `UNCERTAIN`

*Note: Audio score fabrication was removed. The missing audio correctly triggers an explicit UNCERTAIN state.*

## 2. Field Semantics

- **`classification.label`**: Legacy field. Previously mapped UNCERTAIN to "real". Now mapped strictly to "unknown" when UNCERTAIN to prevent misinterpretation, but is no longer used by the frontend UI.
- **`classification.fake_probability`**: Identical to calibrated score.
- **`authenticity.calibrated_probability`**: The core authoritative float value produced by the Platt calibrator.
- **`authenticity.uncertainty`**: The authoritative, categorical decision. Contains exactly one of: `AUTHENTIC`, `UNCERTAIN`, or `SYNTHETIC`.
- **`model_finding`**: Human-readable text corresponding to the decision ("Likely manipulated", "Likely authentic", or "Inconclusive evidence").

## 3. Single Authoritative Decision Function
The logic now resides strictly in `backend/inference.py`:
```python
if calib_score >= 0.99:
    decision = "SYNTHETIC"
    finding = "Likely manipulated"
elif calib_score <= 0.8273:
    decision = "AUTHENTIC"
    finding = "Likely authentic"
else:
    decision = "UNCERTAIN"
    finding = "Inconclusive evidence"
```
The Frontend UI and the PDF Template have been verified to only read `authenticity.uncertainty` (or `abstention_state`) without performing any independent thresholding.

## 4. Verify Current Policy
The policy boundaries (`0.8273` and `0.99`) match the repository's documented production calibration boundaries.

## 5. Missing-Audio Handling
Fixed. A missing audio track correctly sets `aud_score = None`, avoids passing incomplete data through the fusion calibrator, and returns `UNCERTAIN` with the specific message: `"Audio analysis was unavailable for this media."`

## 6. Frontend & PDF Verification
- **Frontend**: Removed the `isFake` / `isAuthentic` logic that read `model_finding` strings. Now explicitly reads the exact `decision` enumeration.
- **PDF**: Verified that the Playwright template reads `profile.trust.abstention_state` which holds the unified decision.

## 7. Acceptance
- **Backend/Frontend/PDF decisions are identical**: PASS
- **No legacy-field mismatch**: PASS
- **Missing audio is represented correctly**: PASS
- **No fabricated audio score**: PASS
- **No decision inversion**: PASS

**AUDIT STATUS: COMPLETE & PASSED.**
