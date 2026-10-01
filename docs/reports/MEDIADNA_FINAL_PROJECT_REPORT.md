# MediaDNA Final Project Report
## Summary
The MediaDNA project successfully delivered an end-to-end multimodal forensic pipeline for deepfake detection. 

## Technical Achievements
1. Diagnosed and bypassed the V14 AVFF "pooling collapse" bug via independent visual and audio specialists (V22.3).
2. Proved observational sensitivity to audio streams through the diagnostic triad.
3. Implemented rigorous data hygiene (Train/Dev/Cal/Test splits).
4. Engineered a production-ready application with probabilistic calibration (Platt) and a clear three-state decision layer (AUTHENTIC / SYNTHETIC / UNCERTAIN).

## Scientific Limitations
While the engineering and diagnostic objectives were met, generalized performance on the locked test set (ROC-AUC 0.601) remains modest. The model serves as a robust proof-of-concept for multimodal forensics but is not yet a guaranteed discriminator for all unseen deepfakes.
