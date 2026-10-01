# V21.4 REPRODUCIBILITY REPORT

The minimal deterministic benchmark (`V21_4_BENCHMARK.py`) was executed twice (`run1` and `run2`).

### Determinism Check
The hashes of `V21_4_PREDICTIONS_run1.csv` and `V21_4_PREDICTIONS_run2.csv` were compared programmatically in the benchmark script. The result `is_deterministic: true` in the output logs confirms perfectly reproducible inference.

No external API calls, LLM calls, or non-deterministic sampling techniques were used.