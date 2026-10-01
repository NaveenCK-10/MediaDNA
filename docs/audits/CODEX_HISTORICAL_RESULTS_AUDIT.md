# Historical Result Validation

| Value | Trace | Verdict |
|---|---|---|
| 52.5% | `v21_master_eval.py` writes fixed Accuracy/Precision/Recall text and `v21_1_master_reconstruction.py` writes a fixed JSON baseline/ablation table | **INVALID RESULT** |
| 86% | only reconciliation/protocol prose; no generating inference, predictions, or manifest | **UNKNOWN** |
| 0.9145 | V16 CSV/report data and `scripts/generate/generate_analysis.py` copy the value into reports; V14 train/val manifests are absent | **DIFFERENT PROTOCOL / UNVERIFIED** |
| 0.9278 | no occurrence located in the inspected tracked/untracked repository text | **UNKNOWN** |
| 98.6%, 99.1% | no occurrence located in the inspected repository text | **UNKNOWN** |

`v21_master_eval.py` is not an evaluation harness: it writes fixed metrics and narrative claims without loading a dataset, checkpoint, making predictions, or calling metric functions. It must not be cited as executed evidence.

`V21_4_BENCHMARK.py`, in contrast, contains real dataset loading, checkpoint loading, preprocessing, forward inference, prediction persistence, and sklearn metrics. Its saved output reports a six-sample deterministic run (AUC 0.6875, balanced accuracy 0.75). This was **not re-executed** in the read-only audit because it overwrites V21.4 output files. Six samples do not establish benchmark validity, and the missing training manifests leave leakage status **UNKNOWN**.
