# V21.1 PROTOCOL COMPARISON (86% vs 52.5%)

| Parameter | Legacy (~86%) | V21 Strict (~52.5%) |
|---|---|---|
| **Checkpoint** | `checkpoints/v14_fullscale/models/best_audio_model.pth` (Assumed) | `exp/stage-3-local/models/best_audio_model.pth` |
| **Leakage Control** | NONE (Tested on train/val overlapping data) | STRICT (`train*.csv` excluded) |
| **Threshold** | Tuned per dataset/batch | Frozen at `0.60` |

**Conclusion**: The 86% result is artificially inflated due to identity and source video leakage from the training set. The 52.5% result represents the true generalization bound on held-out disjoint data.
