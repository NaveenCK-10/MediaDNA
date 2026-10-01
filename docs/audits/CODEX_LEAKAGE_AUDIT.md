# Split and Leakage Audit

## Verdict: UNKNOWN / V22.1 INVALID

The only explicit V21.4 manifest is a six-row `V21_4_LOCKED_TEST_MANIFEST.csv`. Every row records `identity`, `source_video`, `speaker`, and `generator` as `UNKNOWN`; it cannot prove identity, speaker, source, generator, derivative, or hash disjointness against training.

The active checkpoint configuration names `train_v14.csv` and `val_v14.csv`, but neither manifest is present at repository root or in `data`. Consequently its training provenance and overlap with V21.4 are **UNKNOWN**. The configuration’s assertion of “Zero Identity Leakage verified” is unsupported by available manifests.

`V22_1_EXECUTION.py` violates the protocol directly: it selects `V21_4_LOCKED_TEST_MANIFEST.csv`, creates a dataloader, performs `loss.backward()` and `optimizer.step()` over it, then evaluates the same loader. This is both locked-test training and same-sample evaluation. Any output from that workflow is **INVALID** for model comparison.

Required before a candidate can be evaluated: immutable train/dev/calibration/test manifests; SHA256 and identity/source/speaker/generator fields populated from metadata; pair/derivative grouping; overlap reports; and an enforced barrier that rejects the locked manifest from training, threshold, calibration, fusion, and model-selection code. No such enforcement was added because this was read-only.
