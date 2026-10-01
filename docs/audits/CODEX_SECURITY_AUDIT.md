# Security Audit

## Current status: PARTIAL; not production-ready

Observed protections: `/api/analyze` restricts extensions/content type, streams files while enforcing a 500 MB limit, uses server-generated temporary paths, hashes upload bytes, and removes the temporary file after background inference. FFmpeg calls use argument arrays rather than a shell string. Visible NVIDIA credentials are retrieved from environment variables, not hard-coded values.

Gaps: extension/content type are not magic-byte validation; no authenticated/authorized API boundary or rate limit was found; no explicit decode duration/resolution/pixel limit runs before model work; FFmpeg audio extraction has no timeout; GPU work is scheduled without a concurrency semaphore; history persistence is a shared JSON file; and `_inspect_media` calls `ffprobe` by name despite it not being on PATH in this audit environment. No sandbox/resource containment was verified.

`/api/analyze-demo` exposes local fixed dataset paths when enabled. Upload handling needs magic/container verification, bounded FFprobe/FFmpeg subprocesses, pre-decode media limits, queue backpressure, authentication/authorization, rate limits, structured audit logging, and safe persistence before any production claim.
