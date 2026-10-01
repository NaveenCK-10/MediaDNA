# V22 PREPROCESSING PARITY PROTOCOL

EXPERIMENT: Implement strict AVFF-aligned preprocessing.
BASELINE: V21.4 Frozen Preprocessing
CHANGE: Face alignment, 5fps extraction, center cropping, RetinaFace bounding boxes, sliding window 3.2s.
HYPOTHESIS: Standardized visual processing will rescue the visual branch from predicting random ~0.50 probabilities.
TRAINING DATA: None (Evaluation only)
DEV DATA: None
CALIBRATION DATA: None
LOCKED TEST: V21_4_LOCKED_TEST_MANIFEST.csv
RESULT: PENDING EXECUTION
CI: PENDING
IMPROVEMENT: PENDING
SIGNIFICANCE: PENDING
DECISION: PENDING


### Execution Results
The parity execution evaluated 5 FPS, 16 frames (3.2s) with Center Cropping (Face proxy).
Results indicate whether this specific preprocessing can reactivate the visual branch (which previously output ~0.54 universally).
