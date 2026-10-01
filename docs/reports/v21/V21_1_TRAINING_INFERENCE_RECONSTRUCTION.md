# V21.1 TRAINING & INFERENCE RECONSTRUCTION

**Optimizer**: UNKNOWN (No explicit training scripts currently active in root).
**Loss**: Cross Entropy (Logits are processed into probabilities).
**Augmentations**: UNKNOWN.
**Classifier**: Softmax over 2 classes (real/fake).
**Inference Strategy**: `inference.py` loads the model directly onto GPU, computes raw logits, and uses a hard threshold.
