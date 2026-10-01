# Runtime Audit

## Verdict: PARTIAL path validation; no genuine E2E certification

Read-only validation successfully imported `backend.main`; the configured FFmpeg executable exists. This verifies Python import wiring only. It does not load the model or decode media, because startup loads the 748 MB checkpoint and a genuine API test creates history/job/output state.

The nominal path is upload → temp file + SHA256 → background job → audio/video decode → model → auxiliary modules → JSON history → job result → report endpoint. The code preserves the same asset SHA256 for identical input bytes and assigns a new `run_` job ID per request. It does not persist a verified checkpoint hash or manifest hash into returned `MediaDNAProfile` records.

No read-only execution was available for all four RVRA/RVFA/FVRA/FVFA cases, no-audio, low-resolution, unsupported, and corrupt media without creating histories/results. Therefore latency, p95, VRAM/RAM/CPU, throughput, failure rate, report persistence, and complete case lifecycle are **UNKNOWN**.

The backend should not be claimed as runtime-verified until a non-mutating/staged test harness captures those facts and validates cleanup and error paths.
