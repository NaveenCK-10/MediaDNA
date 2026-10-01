# V21.1 LEAKAGE REPORT

**Status**: CRITICAL LEAKAGE DETECTED (Historical)
The `final_baseline_eval.py` script attempts to mitigate leakage by explicitly explicitly excluding `train*.csv` and `val*.csv` paths from `FakeAVCeleb_v1.2` scanning. 
However, generator and speaker identity leakage is inherently prevalent if evaluating on subsets of the same distribution without completely disjoint identity mappings.

**Action Required**: A genuinely disjoint "locked_test" subset must be physically separated.
