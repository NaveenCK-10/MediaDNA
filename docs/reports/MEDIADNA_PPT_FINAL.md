# MediaDNA: Final Presentation

## Slide 1: The Problem
- Deepfakes are increasingly multimodal.
- Existing tools rely heavily on visual-only artifacts.

## Slide 2: The Solution
- MediaDNA: A dual-stream specialist architecture.
- Late fusion to prevent modality collapse.

## Slide 3: Diagnostic Triad
- Silencing and replacement tests confirm audio sensitivity.
- Evidence is consistent with multimodal contribution.

## Slide 4: System Architecture
- React Frontend -> FastAPI Backend -> PyTorch Inference.
- 3-State Decision: Authentic, Synthetic, Uncertain.

## Slide 5: Limitations & Future Work
- Model calibration reveals high false-positive rates on pristine data.
- Future work: Adaptive thresholding and larger, balanced training sets.
