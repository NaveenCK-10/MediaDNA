# V22 AUDIO SHORTCUT REPORT

EXPERIMENT: Controlled Paired A/V Substitution.
BASELINE: V21.4 Frozen Benchmark
CHANGE: Substitute audio track for paired visual frames (Silence, Target Speaker, Imposter Speaker, Synthetic).
HYPOTHESIS: The model is acting entirely as a voice-anomaly detector and ignoring visual manipulation.
LOCKED TEST: V21_4_LOCKED_TEST_MANIFEST.csv
RESULT: PENDING (Supported empirically by V22_MODALITY_ABLATION.csv where V-Only scores collapse to ~0.54)
DECISION: PENDING (High Priority Fix Required for Visual Branch)


### Executed Paired Substitution
- FakeVideo + RealAudio (Original Score): 0.5107
- FakeVideo + FakeAudio (Substituted Score): 1.0000
- RealVideo + FakeAudio (Original Score): 1.0000
- RealVideo + RealAudio (Substituted Score): 0.5015

**CONCLUSION**: The visual manipulation state (Real vs Fake) has effectively zero impact on the final prediction. The final classification score is dictated entirely by whether the Audio track is Real or Fake, proving a causal audio shortcut.