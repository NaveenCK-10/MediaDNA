# V22.1 MODEL E: MODALITY-SPECIFIC HEADS

## Objective
Explicitly penalize the model for failing on single-modality manipulations (e.g. FakeVideo+RealAudio).

## Architecture Change
Add a `Video_Fake` head and an `Audio_Fake` head alongside the `AV_Fused` head.
Train with a multitask objective.

## Results
*PENDING EXECUTION*