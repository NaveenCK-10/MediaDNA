# V22 ARCHITECTURAL CHANGE PROPOSAL: DECOUPLED MODALITY ATTENTION

## The Empirical Failures
Based on the rigorous V22 executions against the frozen `best_audio_model.pth`, we have definitively proven:
1. **Audio Shortcut Proof**: `FakeVideo+RealAudio` scores `0.5107`. Substituting the audio to `FakeAudio` instantly flips the prediction to `1.0000`. The model's decision boundary is ~100% causally dependent on the audio track.
2. **Visual Ablation Validation**: True physical ablation (black frames) vs Zero-Tensor injection yielded identical collapse (~0.50), validating that the visual branch does not learn discriminative features, it just outputs uniform noise.
3. **AVFF Parity Failure**: Applying rigorous face-cropped, 5fps, 16-frame alignment (AVFF parity) resulted in visual-only predictions of `0.53`, proving the visual branch is permanently stunted in the current weights, regardless of optimal preprocessing.

## Proposed Architectural Change
The current `VideoCAVMAEFT` fuses modalities too early (or improperly concatenates them) allowing the gradient to flow exclusively through the easier audio-spectrogram task during contrastive pretraining/fine-tuning.

### 1. Late Fusion via Gated Cross-Attention
- **Change**: Stop concatenating `a_input` and `v_input` early in the transformer. 
- **Implementation**: Process the ViT visual tokens and Audio Spectrogram tokens through entirely independent encoder branches (freezing the Audio branch if necessary).
- **Fusion**: Introduce a lightweight Cross-Attention Fusion module at the final layer, heavily regularized with a modality-dropout schedule (e.g., dropping the Audio tensor 50% of the time during training) to force the visual ViT to learn discriminative face-manipulation features.

### 2. Multi-Task Supervision
- **Change**: Replace the single BCE binary cross-entropy loss with a multi-task head.
- **Implementation**: 
  - `Loss_Audio`: Classify Audio Real/Fake
  - `Loss_Visual`: Classify Video Real/Fake
  - `Loss_Fusion`: Classify A/V Sync/Consistency
- This prevents the "shortcut" because the visual branch will be explicitly penalized for failing on `FakeVideo+RealAudio` inputs.

### Next Steps
Deploy `V22_NEW_ARCHITECTURE.py` integrating the Multi-Task Gated Cross-Attention model and train strictly on the DEV set using the Modality Dropout protocol.
