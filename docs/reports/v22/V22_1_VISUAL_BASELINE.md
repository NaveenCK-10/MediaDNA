# V22.1 MODEL B: INDEPENDENT VISUAL BASELINE

## Objective
Determine whether visual information itself is learnable on this dataset independent of audio.

## Pipeline
Video -> Face detection/alignment -> Face crop -> Visual backbone (ViT) -> Temporal aggregation -> Binary classifier.

## Protocol
- **Train**: TRAIN split only.
- **Selection**: DEV split only.
- **Evaluation**: V21.4 Frozen Locked Test Manifest.

## Results
*PENDING EXECUTION*