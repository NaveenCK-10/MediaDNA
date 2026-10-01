# V22.3 Calibration Report (Platt Scaling)

**Date:** 2026-09-28T18:15:38.986220
**CAL split samples:** 2195
**Method:** Platt scaling (logistic regression)

## Platt Coefficients

- Coefficient: 0.070894
- Intercept: 3.703216

## Calibration Metrics

| Split | Brier (raw) | ECE (raw) | Brier (calibrated) | ECE (calibrated) |
|-------|-------------|-----------|--------------------|-----------------|
| CAL | 0.0468 | 0.1568 | 0.0223 | 0.0001 |
| DEV | — | — | 0.0183 | 0.0041 |

## Decision Policy

- **AUTHENTIC**: calibrated probability < 0.3
- **UNCERTAIN**: 0.3 ≤ calibrated probability < 0.7
- **SYNTHETIC**: calibrated probability ≥ 0.7

## DEV Decision Distribution

- AUTHENTIC: 0
- UNCERTAIN: 0
- SYNTHETIC: 1982
