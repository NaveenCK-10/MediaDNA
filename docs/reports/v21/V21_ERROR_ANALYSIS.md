# V21 Error Analysis

## Top Failure Modes
- **FakeVideo + RealAudio**: OpenAVFF relies heavily on audio spectrograms. When real audio is laid over deepfake video, the false negative rate spikes severely.
- **High Compression**: Eliminates high-frequency spatial anomalies, blinding the visual branch.

## Identity Leakage
Analysis indicates that models frequently overfit to dominant identities in the dataset. External datasets are strictly required to resolve this.