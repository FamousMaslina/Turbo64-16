# Local validation results

Date: 2026-10-07. Windows 11, Ryzen 7950X, 64 GB RAM, RTX 4080, driver 610.74. Supplied source: llama.cpp-b11475 archive, no Git metadata. Release build: MSVC 19.44.35228, CUDA 12.8, sm_89, native CPU instructions.

## Full server configuration

Mode 2, same model and user settings: context 65536, GPU layers all, CPU MoE 47, decode threads 16, batch threads 24, batch 4096, microbatch 1024, Flash Attention on, mmap, one slot, Jinja, reasoning on. Validation used loopback port 5558 and disabled startup warmup. User launcher uses the original port 5559 and default warmup behavior.

Each request had 1024 input tokens with `cache_n=0`. The prompt was repeated inference-related prose through `/completion`, not a chat-template quality evaluation.

| Run | Prompt tokens | Prefill | Generation | Prompt time |
| --- | ---: | ---: | ---: | ---: |
| First | 1024 | 20.56 tokens/s | 1.74 tokens/s, 8 predicted tokens | 49.80 s |
| Repeat | 1024 | 18.44 tokens/s | 2.18 tokens/s, 32 predicted tokens | 55.53 s |

Observed total dedicated device usage from nvidia-smi: approximately 13,108-13,114 MiB (12.8 GiB), including desktop applications. This is not a per-process measurement. Task Manager shared GPU memory was not measured. Process working set reached approximately 54 GiB, including file-backed pages; this is not an extra 54 GiB allocation on top of file caching.

Raw results: `results-turbo2-64k.json`, `results-turbo2-64k-repeat.json`, `benchmark-64k.log`, `benchmark-64k-repeat.log`, `server-test.stderr.log`.

User-reported previous stock result: 8.35 tokens/s prefill and 3.83 tokens/s generation. Turbo's observed prefill is 2.21-2.46 times that number, but the original prompt, cache state, run length, and binary were not reproduced. This is a comparison to a reported reference, not a controlled speedup claim. The generation numbers are lower and need matched A/B testing before deciding whether this tradeoff is useful.

The 64K KV allocation was tested; attention with 64K populated tokens was not. Larger microbatches, mode 1 versus mode 2, cold versus warm disk behavior, and normal chat output remain for user testing. No claim is made that staging independently accounts for the improvement: retaining file-backed experts and changing the scheduler's prefill placement are substantial parts of the patch.

## Short bench sanity check

Available stock binary: b11447, CUDA 12.4, dynamic zen4 CPU backend. Local patched binary: b11475, CUDA 12.8, MSVC native CPU backend. Both used pp128, no warmup, one repetition, GPU layers 99, CPU MoE 47, threads 24, batch 4096, microbatch 1024, Flash Attention, mmap.

| Binary | pp128 |
| --- | ---: |
| Available stock b11447 | 1.696 tokens/s |
| Turbo64-16 mode 2 | 3.845 tokens/s |

These are different source/compiler configurations and sequential runs with uncontrolled file caching. They are sanity checks, not a rigorous performance comparison, and llama-bench did not allocate the server's full 64K context. Raw output: `stock-pp128.json`, `turbo-pp128.json`.

## Correctness and build

- Release build passed for llama-server, llama-bench, test-backend-ops, and test-turbo64-16.
- Mode 1 transfer validation: 10/10 Q2_K/MXFP4 selection cases passed byte-exactly.
- Mode 2 transfer validation: 10/10 Q2_K/MXFP4 selection cases passed byte-exactly.
- CUDA MUL_MAT_ID versus CPU: 77/77 selected tests passed. Existing tests' numerical tolerances apply.
- Windows PrefetchVirtualMemory support is present in the built binary.
- Python scripts compile; patch replay and reversal are checked against the supplied originals.

This does not establish full-model perplexity or token-identical generation. The patch preserves selected experts, weight bytes and padding, but changing CPU/CUDA execution can change floating-point rounding.

Mode 0 preserves the original source behavior in the same compiled binary and is the preferred next A/B baseline. See README.md for launch and measurement commands.

## Bespoke LAN server validation

The dedicated executable is `turbo64-16/bin/llama-server.exe`; the original server entry point is retained separately under `build-turbo64-16/bin/`. The dedicated main adds the TURBO banner and defaults for debug logs, timestamps, metrics, slots, performance reporting, and a five-second Windows hardware reporter. Expert transfer/graph timers are enabled by `TURBO64_16_DIAGNOSTICS=1` and are off unless requested in other tools.

The LAN listener was verified at `0.0.0.0:5559`. Access through the PC's Ethernet address `192.168.0.234:5559` returned HTTP 200 for `/health`, `/v1/models`, `/props`, `/slots`, and `/metrics`, and an OpenAI-compatible `/v1/chat/completions` request returned an assistant message with normal usage fields. The metrics output included Prometheus type declarations. This test originated on the server PC via its LAN IP; access from another LAN device and Windows Firewall allowance were not tested or modified.

Mode 2's byte-exact transfer tests also passed with diagnostic timing enabled. Console and saved-file hardware reports were observed. Raw evidence: `api-smoke.json`, `api-smoke-check.log`, `api-smoke.stderr.log`, `server-version.log`, `test-stream-diagnostics.log`, and files under `bin/logs/`. The banner and API schema checks do not require repeating GPU kernel correctness tests because no kernels or response serializers were changed.

Diagnostics add logging overhead. The prefill figures above were measured before enabling debug logging and hardware reporting.

An uncached 128-token native completion additionally exercised the instrumented GPU expert path: the main 124-token graph reported 138 expert tensors, 18,330 selected matrices, 58,049.19 MiB requested uploads, and 8,720 chunks. Measured host categories were 7,662.88 ms prefetch calls, 27,504.50 ms staging (including possible page-fault stalls), 400.89 ms staging-buffer waits, and 137.90 ms enqueue calls. This demonstrates actual counters on the model; these are not physical disk/GPU-bandwidth measurements. Raw evidence: `results-diagnostics-pp128.json` and `diagnostics-prefill-check.log`.

## Content-free logging update

The bespoke executable now installs a one-way metadata-only logging filter before server initialization. Unapproved log formats are dropped before log formatting/queueing; preformatted backend records use an explicit list of technical producers. Structured content dumps are disabled, chat parser debug output is suppressed, and the prompt-file writer is blocked. Verbosity and JSONL settings cannot remove this policy. The benchmark helper no longer saves generated text.

The dedicated regression passed at maximum verbosity with JSONL and parser debugging requested. Synthetic message canaries were absent from both console and file output; hardware, graph, HTTP status/size, and prompt timing counters remained. Prompt-file capture arguments were rejected before creating a directory. Evidence: `private-log-regression.console.log`, `private-log-regression.jsonl`, and `private-prompt-option.log`.

An API check using the real model passed for normal chat (HTTP 200), streaming chat (HTTP 200 with completed SSE), and malformed JSON (HTTP 400). With `--verbose --log-jsonl`, request canaries and generated message/reasoning text were absent from terminal captures and the saved log. Hardware, HTTP metadata, and prompt timing records remained. The test report stores status codes and token usage only, without chat contents: `privacy-api-check.json`.

Earlier API smoke tests and older log/result files described above predate this policy and have not been scrubbed. The updated policy applies to the bespoke executable, including Turbo mode 0. Other binaries retain their normal logging unless they explicitly install a metadata-only filter.
