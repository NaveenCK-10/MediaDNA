# V22.1 MODEL D: AV MODEL + MODALITY DROPOUT

## Objective
Prevent the model from over-relying on the audio shortcut during training.

## Protocol
- Start from a defensible pretrained configuration.
- Randomly drop Audio input for a controlled percentage during training.
- Randomly drop Visual input for a controlled percentage during training.
- Compare dev-set values (10%, 25%, 50%) to select the optimal hyperparameter. Do not arbitrarily choose 50%.

## Results
*PENDING EXECUTION*