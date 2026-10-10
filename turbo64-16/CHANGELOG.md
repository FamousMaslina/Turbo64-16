# Turbo64-16 v0.2.1 - 2026-10-10

- Added `--prefill-kv`: 65536 context, GPU Q8 K/V, batch 16384, microbatch 9216, and the existing MiMo model and thread settings.
- Available through the packaged executable and `run-server.cmd`, with native CLI overrides. Existing profiles, no-argument startup, and CUDA graphs are unchanged.
- Named-preset validation with the complete benchmark document and 512 output tokens measured 101.75 tokens/s prefill and 3.12 tokens/s decode. Input/output hashes matched the earlier explicit configuration; native alias/negative overrides and privacy regression passed.

# Local prefill update - 2026-10-10

- Added `--turbo-prefill2` with the previous batch/microbatch 4096 and GPU F16 KV settings. Existing profiles remain available.
- Restored-profile medians over two complete document/512-token output requests: prefill 38.59 tokens/s, output 2.78 tokens/s. Later output intervals reached 3.1-3.3 tokens/s; complete output averages were below 3.
- Measured the full 8564-token document before changing settings: default prefill 15.01 tokens/s, decode 1.51 tokens/s; previous prefill profile 42.55 tokens/s.
- Updated `--turbo-prefill` to batch 16384, microbatch 9216, and CPU F16 KV, retaining 65536 context capacity and CUDA graphs.
- Rebuilt-profile prefill: 90.48-112.66 tokens/s, median 101.57 (6.77x the starting default); median decode 1.54 tokens/s. File-cache state was uncontrolled.
- Added complete-document benchmarks with cache/count checks and incremental median reports. Build, byte-exact transfer, privacy, and repeated model checks passed.

# Turbo64-16 V0.2

- Prefill on the 7,482-token workload: 14.19 to 47.64 tokens/s (3.36x); prompt time 527.21 to 157.06 seconds.
- Requested expert uploads: 665.00 to 195.16 GiB; bounded asynchronous read-ahead overlaps staging.
- Added `--turbo-prefill`, `--turbo-decode` and `--turbo-balanced`, with native CLI overrides.
- Decode: 3.18 to 3.11 tokens/s; output lengths differ. CUDA graphs unchanged.
- Source-only release. Transfer, backend, profile and privacy checks passed.
