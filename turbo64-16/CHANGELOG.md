# Turbo64-16 V0.2

- Prefill on the 7,482-token workload: 14.19 to 47.64 tokens/s (3.36x); prompt time 527.21 to 157.06 seconds.
- Requested expert uploads: 665.00 to 195.16 GiB; bounded asynchronous read-ahead overlaps staging.
- Added `--turbo-prefill`, `--turbo-decode` and `--turbo-balanced`, with native CLI overrides.
- Decode: 3.18 to 3.11 tokens/s; output lengths differ. CUDA graphs unchanged.
- Source-only release. Transfer, backend, profile and privacy checks passed.
