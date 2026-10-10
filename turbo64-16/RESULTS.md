# Local validation results

Date: 2026-10-07. Windows 11, Ryzen 7950X, 64 GB RAM, RTX 4080, driver 610.74. Supplied source: llama.cpp-b11475 archive, no Git metadata. Release build: MSVC 19.44.35228, CUDA 12.8, sm_89, native CPU instructions.

## 2026-10-10: restored prefill2 profile

Added `--turbo-prefill2` with the settings used before the CPU-KV prefill update: batch 4096, microbatch 4096, GPU F16 KV, 65536 context capacity, CPU MoE 47, 16 decode/24 batch threads, Flash Attention, mmap, one slot, Jinja, reasoning, mode 2, and asynchronous 128 MiB read-ahead. The existing prefill, decode and balanced profiles remain available; no-argument startup still selects decode. CUDA graphs and kernel sources were not changed.

The rebuilt packaged executable ran two complete requests using `sample-doc.txt` and `sample-prompt.txt`, the same chat template and 8564-token prompt hash as the earlier tests, greedy sampling, seed 1234, and 512 generated tokens per request. Diagnostics and startup warmup were disabled. Both requests processed all 8564 tokens with `cache_n=0` and produced identical 512-token response hashes: `6a860c627aabf4affb52b1196f2e5ff6826cbb9c15a565f4fe3e561edc46c9d1`.

| Restored profile | Prefill tokens/s | Prompt seconds | Output tokens/s | Generated tokens |
| --- | ---: | ---: | ---: | ---: |
| First request | 39.13 | 218.881 | 2.72 | 512 |
| Second request | 38.05 | 225.068 | 2.85 | 512 |
| Median | 38.59 | 221.974 | 2.78 | 512 |

These runs support approximately 40 tokens/s prefill. The complete output averages were below 3 tokens/s, while later generation intervals reached approximately 3 tokens/s. Wall-clock `/slots` observations measured 3.08 tokens/s across tokens 147-441 in the first request (95.50 seconds), and 3.31 tokens/s across tokens 204-444 in the second (72.61 seconds). These interval measurements exclude the slower initial generation and must not replace the complete native timing averages. The longer 512-token output differs from the earlier 64-token benchmarks; output rates are not a matched before/after speedup comparison. Windows file-cache state was uncontrolled, and 64K capacity was allocated while only the benchmark document and generated tokens were populated.

Logs confirmed `profile=prefill2`, both batch sizes at 4096, context 65536, GPU F16 global/SWA KV allocations of 1440.00/828.75 MiB, Flash Attention enabled, and 128 MiB asynchronous read-ahead. Build/package, launcher forwarding/help, conflicting-profile rejection, preserved-profile/default selection, ASCII/whitespace and patch checks passed. The benchmark server was stopped after testing. Raw evidence: `results-prefill2-restored-pp8564-tg512.json`, `prefill2-restore.stderr.log`, and `prefill2-restore-progress.json`.

Run the restored profile with `turbo64-16\run-server.cmd --turbo-prefill2`. For the benchmark, set `TURBO64_16_DIAGNOSTICS=0` and use `benchmark.py --document turbo64-16/sample-doc.txt --prompt turbo64-16/sample-prompt.txt --repetitions 2 --generate 512 --label prefill2-restored`.

## 2026-10-10: complete document prefill

Measured the starting packaged executable before changing performance settings, using the complete `sample-doc.txt` followed by `sample-prompt.txt` in one user message. The server's `/apply-template` and `/tokenize` endpoints produced 8,564 tokens; the exact token hash was `df4676a58139a24751179b34211264b6cad603816c8aee7503abe985bb4999a3` in every completed run. `/completion` used 64 greedy generated tokens, seed 1234, and `cache_prompt=false`. All completed runs processed exactly 8,564 tokens with `cache_n=0`.

All configurations used the same local MiMo GGUF, 65,536 context capacity, one slot, F16 keys and values, Flash Attention, mmap, GPU layers all, CPU MoE 47, 16 decode/24 batch threads, mode 2, and asynchronous 128 MiB read-ahead. Diagnostics and startup warmup were disabled. The initial default-profile run used info verbosity 3; later runs used 4 to capture technical configuration and timing logs. CUDA graphs, kernels, model weights, and expert routing code were not modified.

| Initial configuration | Logical batch | Microbatch | KV placement | Prefill tokens/s | Prompt seconds | Decode tokens/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Starting default/decode profile | 4096 | 1024 | GPU F16 | 15.01 | 570.715 | 1.51 |
| Previous prefill profile | 4096 | 4096 | GPU F16 | 42.55 | 201.254 | 1.63 |
| Larger GPU-KV microbatch | 16384 | 6144 | GPU F16 | 58.10 | 147.400 | 1.86 |
| Complete-document microbatch | 16384 | 9216 | CPU F16 | 113.08 | 75.737 | 1.28 |

The initial 9216 result was 7.54x the starting default and 2.66x the previous prefill profile. Prompt time fell 86.7% versus the starting default. Decode was about 15% slower than the starting default in this sample; this change prioritizes prefill only. CPU KV placement is not KV quantization. It frees persistent GPU KV storage for a larger compute batch, with additional CPU/GPU transfers and host memory use. The prefill consists of one 8560-token expert-transfer pass plus a separate four-token checkpoint tail. The previous 4096 profile used 4096, 372 and 4092-token passes plus the same tail.

After rebuilding, a fresh process was launched with `--turbo-prefill --host 127.0.0.1 --port 5558 --no-warmup`, without batch or KV overrides. Allocation logs confirmed the intended batch, microbatch, context, and CPU F16 KV settings. Both full-document repetitions had `cache_n=0` and the same response hash as the earlier 9216 run, `7686bd0702f7767cfddc8c6c0979fd2168018af7ec7abef6bde720c27a9ca421`.

| Rebuilt named profile | Prefill tokens/s | Prompt seconds | Decode tokens/s |
| --- | ---: | ---: | ---: |
| First request | 112.66 | 76.018 | 1.48 |
| Second request | 90.48 | 94.647 | 1.61 |
| Median of these two requests | 101.57 | 85.333 | 1.54 |

The repeated profile's median prefill rate was 6.77x the starting default and 2.39x the initial previous-prefill run. Median prompt time fell about 85.0% versus the starting default. The approximately 20% rate variation between requests is material; the initial 113.08 result is not a guaranteed sustained rate. Decode was approximately unchanged versus the starting default in these short samples and was not optimized. Evidence: `results-document-prefill-final.json` and `document-prefill-final.stderr.log`.

A final fresh-process run used `--turbo-prefill --batch-size 4096 --ubatch-size 4096 --kv-offload` to restore the previous prefill configuration on the rebuilt executable. Logs confirmed both batch sizes were 4096 and both KV allocations were GPU F16 with 65536 context capacity. It measured 41.27 tokens/s prefill in 207.513 seconds and 1.99 tokens/s decode, with no cached tokens and the same response hash as the initial previous-profile run. The previous profile's two-run medians were 41.91 tokens/s prefill and 1.81 tokens/s decode. The updated preset was 2.42x faster in prefill, with about 15% lower decode throughput than the previous prefill profile. These comparisons remain subject to file-cache and output differences. Evidence: `results-document-ub4096-repeat.json` and `document-ub4096-repeat.stderr.log`.

The 256 MiB read-ahead experiment was interrupted when VS Code closed, before generation and the final JSON report completed. Its main 8560-token block took 134.68 seconds, versus 71.13 seconds with 128 MiB read-ahead. This partial timing does not supply a final prefill/decode rate; 256 MiB was rejected and the 128 MiB default was retained. Previously completed JSON reports and source edits survived the interruption.

Observed total dedicated GPU memory including desktop applications was about 13,100-13,300 MiB for the default, 14,500 MiB for the previous prefill profile, and 15,500-15,700 MiB for the larger configurations. The final CPU-KV profile allocates 1440.00 MiB global KV and 1803.75 MiB sliding-window KV in CPU RAM, a 6029.11 MiB CUDA compute buffer, and a 2083.91 MiB CUDA host compute buffer. Dedicated/shared memory per process and peak memory were not measured. Desktop GPU use limits headroom.

Windows file caching was not reset or controlled between runs. These are sequential local observations, not a controlled speedup guarantee. Only 8564 prompt tokens were populated; the full 64K capacity was verified in `/props` and allocation logs, but performance at 64K populated tokens was not tested. Initial default and previous-prefill response hashes matched over 64 tokens; 6144 and 9216 hashes differed. Full-model perplexity, final-answer quality, and token-equivalent outputs across microbatches are not established.

The new `--turbo-prefill` profile supplies batch 16384, microbatch 9216, `--no-kv-offload`, and explicit F16 KV. Decode, balanced, and no-argument settings retain their previous defaults. Native aliases and positive KV-offload options can replace profile defaults, as verified by the final comparison run. Rebuilding and packaging passed; all 14 mode-2 transfer cases passed byte-exactly, and the metadata-only logging regression passed. Conflicting profiles were rejected and the launcher forwarded profile help correctly. The Python benchmark now supports paired `--document` and `--prompt` paths, rejects a context other than one 65536-token slot, and saves both median prefill and decode rates after each completed request without saving conversation text. Python syntax and changed-file ASCII/whitespace checks passed; the CI-pinned `ty` checker was unavailable in the local Python environment and could not be installed from the configured package index. Benchmark servers were stopped after validation.

Initial raw reports: `results-document-baseline-ub1024.json`, `results-document-ub4096.json`, `results-document-ub6144.json`, and `results-document-ub9216-cpukv.json`. Configuration and timing evidence is in the corresponding `document-*.stderr.log` files. All artifacts remain local.

## V0.2: user 7,482-token workload

The user reran the same prompt workload with the V0.2 prefill profile. Both logs report 7,482 processed prompt tokens and a 65,536-token context. This is a single before/after observation; initial Windows file-cache state and other runtime conditions were not controlled, and the logs do not contain prompt hashes.

| Metric | Previous Turbo run | V0.2 prefill profile |
| --- | ---: | ---: |
| Prefill rate | 14.19 tokens/s | 47.64 tokens/s |
| Prompt processing | 527.21276 s | 157.05743 s |
| Decode rate | 3.18 tokens/s | 3.11 tokens/s |
| Generated tokens | 663 | 421 |
| Requested expert uploads | 665.00 GiB | 195.16 GiB |
| Large staged prefill graphs | 8 | 2 |
| Host staging | 386.04 s | 131.82 s |
| Prefetch calls / worker time | 93.76 s | 39.33 s |
| Host wait for read-ahead task | synchronous calls | 7.32 s |
| Staging-buffer waits | 2.61 s | 0.48 s |

Observed prefill throughput increased 3.36x (+235.7%); prompt time fell 70.2%, saving 370.16 seconds. Requested expert uploads fell 70.7%. New prefetch worker time overlaps staging and must not be added to it; the Windows hint completing does not guarantee page residency. These counters are not physical disk measurements.

Decode rate was approximately 2.3% lower. The generated lengths differ, so generation duration and total request duration are not fair comparisons. CUDA graphs remain unchanged. The new log confirms `n_ubatch=4096` and `async=1`; actual staged batches were 3386 and 4092 tokens, with four further prompt tokens handled separately.

CUDA compute buffers increased from 1490.54 to 2722.17 MiB, sliding-window KV buffers from 243.75 to 828.75 MiB, and CUDA host compute buffers from 194.55 to 802.20 MiB. Model buffers and full-attention KV allocation were unchanged. RAM utilization still reached 99%. Shared GPU allocation alone does not establish device-memory spill.

Local log references: `turbo-17913872839014607.log` and `turbo-17914001968799851.log`. Raw logs are not included in the source release.

## Read-ahead update and named profiles

The packaged server now supports `--turbo-prefill`, `--turbo-decode` and `--turbo-balanced`. They select microbatches 4096, 1024 and 2048 respectively; all other model settings remain the same. A custom native CLI option replaces the corresponding profile default. No-argument startup retains the 1024 decode profile. CUDA graphs and CUDA kernel sources are unchanged.

Matched synthetic tests used fresh server processes, 4096 prompt tokens, 128 greedy generated tokens, seed 1234, 64K allocated context, no warmup, and diagnostics disabled. The `--workload flash-transfer-v2` prefix produced identical prompt-token hashes; all five cases below also produced identical response hashes. Each configuration ran once in the listed order, without resetting the Windows file cache. These are exploratory measurements rather than controlled repeated speedup estimates.

| Build / configuration | Prefill tokens/s | Decode tokens/s | Prompt seconds |
| --- | ---: | ---: | ---: |
| Preserved pre-update mode 2, microbatch 1024 | 25.97 | 5.84 | 157.69 |
| New asynchronous read-ahead, microbatch 1024 | 30.96 | 7.65 | 132.29 |
| New asynchronous read-ahead, microbatch 2048 | 44.37 | 4.52 | 92.32 |
| New asynchronous read-ahead, microbatch 4096 | 74.16 | 4.25 | 55.23 |
| Experimental per-matrix CPU prefetch, microbatch 4096 | 75.68 | 2.82 | 54.12 |

The 4096 case delivered approximately 2.85x the prefill rate of the matched pre-update run. The CPU prefetch experiment slowed generation and was removed from the final source and build. Smaller microbatches showed faster generation in this sample; file-cache residency can contribute, and the decode profile name does not guarantee a fixed speedup. The balanced profile offers the intermediate microbatch and memory footprint.

The 4096 profile used approximately 15,026-15,065 MiB total dedicated GPU memory in observed device samples, including desktop applications. One process sample showed about 12.94 GiB dedicated and 0.92 GiB shared GPU allocation. Shared allocations include CUDA host buffers; these samples do not isolate memory spill or establish peak memory use. More desktop GPU activity can reduce available headroom. Model process working sets remained around 53 GiB.

Raw matched reports: `results-matched-baseline.json`, `results-matched-async1024.json`, `results-matched-async2048.json`, `results-matched-async4096.json`, and the rejected `results-matched-async4096-cpuprefetch.json`. Pre-update binaries are retained in `baseline-log-build/`, with hashes in `baseline-build-manifest.json`. The synthetic prose prompt differs from the user's original workload; the user's later 7,482-token workload comparison is recorded above.

Transfer validation passed all 14 Q2_K/MXFP4 selections in mode 1, mode 2 with synchronous 128 MiB windows, and mode 2 with asynchronous 32 and 128 MiB windows. Cases cover fragmented selections spanning window boundaries, MMQ padding, the 31/32-token dispatch boundary, repeated uploads, buffer reuse and destruction. CUDA expert matmuls versus CPU passed all 77 selected backend cases. Metadata-only logging regression passed. These checks do not establish full-model perplexity or equivalence for all inputs.

Named-profile integration checks used the actual packaged executable and a shared 3072-token prompt with diagnostics enabled. All three loaded the intended model with 64K context and one slot, respected their microbatch limits, used asynchronous read-ahead, and produced identical greedy response hashes over 16 generated tokens. The largest staged token counts were 3068 (prefill), 1024 (decode) and 2048 (balanced). These short generation checks establish behavior rather than decode performance rankings. Evidence: `profile-model-check.json`, `results-profile-prefill.json`, `results-profile-decode.json`, and `results-profile-balanced.json`.

A separate model run with default warmup verified long native aliases overriding the prefill preset: `--ctx-size 8192`, `--parallel 2`, `--ubatch-size 512`, and `--no-jinja`. The server reported two slots with 4096 context per slot, and a 600-token completion staged no more than 512 tokens per microbatch. Profile help, conflicting-profile rejection, argument-value preservation and the launcher forwarding path also passed. Evidence: `profile-overrides-check.json`, `profile-cli-check.log`, and `launcher-help-check.log`.

## Full server configuration

### User-reported long prompt

A later user-reported Turbo64-16 run processed a 7,482-token prompt in approximately 527 seconds with a configured 65,536-token context. Prefill averaged 14.19 tokens/s; generation averaged 3.18 tokens/s over 663 generated tokens.

Compared with the earlier reported stock reference of 8.35 tokens/s prefill and 3.83 tokens/s generation, this is approximately 1.70x prefill (+70%) and 17% lower generation. This is not a controlled A/B comparison. The subsequently supplied `turbo-17913872839014607.log` confirms the long-run timings and configuration; the binary revision and initial file-cache state remain unverified. The 65K figure describes configured context capacity, not a fully populated context.

The supplied log records eight staged prefill graphs with 665.00 GiB requested expert uploads, 386.04 s host staging, 93.76 s prefetch calls, 2.61 s staging-buffer waits, and 1.27 s enqueue calls. RAM availability during most of prefill was approximately 0.26 GiB at 99% utilization. These counters do not measure physical disk reads. Decode's 663 graph submissions had a 200.75 ms median; 14 submissions above one second consumed 51.48 s in total, with a maximum of 7.70 s.

### Local 1,024-token tests

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

## 2026-10-10: KV/context matrix (tests only)

Completed 18 uncached full-document requests across 12 requested variants and one existing CPU-KV reference. Every request used sample-doc.txt plus sample-prompt.txt, the same chat template and 8564-token prompt hash, with 32K or 64K allocated capacity. The initial matrix generated 64 tokens; five confirmations generated 512 tokens. Diagnostics were enabled consistently in the fresh baselines and variants. CPU-only commands used -ngl 0 --device none --no-op-offload, with zero GPU layers/compute buffers confirmed. No profiles, server code, or CUDA graphs were changed.

| 512-token confirmation | Batch / microbatch | Prefill t/s | Decode t/s | Minimum sampled free VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| GPU KV, 64K F16 | 4096 / 4096 | 40.16 | 2.74 | 2153 |
| GPU KV, 64K Q8 | 4096 / 4096 | 40.88 | 2.76 | 3217 |
| GPU KV, 64K Q8 | 16384 / 9216 | 117.68 | 3.12 | 412 |
| GPU KV, 32K Q8 | 16384 / 9216 | 118.85 | 3.03 | 1368 |
| CPU-only, 32K F16 | 4096 / 4096 | 20.08 | 1.91 | 15204 |

64K Q8 at the original batch saved about 1064 MiB of sampled dedicated headroom with approximately unchanged throughput. Both larger Q8 GPU cases reached about 118 t/s prefill and 3.0-3.1 t/s decode; 32K Q8 left more free VRAM than 64K Q8 at that batch. CPU-only F16 prefill was faster than CPU-only Q8 in the initial matrix, but the entire model remains disk-backed because it exceeds RAM. Host staging consumed roughly 77-82% of the confirmed GPU prefill time, and all cases reached 99% RAM load. Raw logs retain per-tensor transfers, hardware observations, disk/page-input counters, and dedicated/shared GPU allocations for later investigation; no bottlenecks were fixed.

Windows file cache and background activity were uncontrolled; repeated prefill varied materially. Capacity was allocated but only the document and generated tokens were populated. Output hashes differ across some variants, and output quality was not evaluated. All test servers were stopped. Full tables, exact commands, limitations, validation, and matching raw-log folders: [local benchmark summary](bespoke-tests/2026-10-10-kv-context/SUMMARY.md).

## 2026-10-10: --prefill-kv preset validation

Added the requested preset using the tested large-batch GPU configuration: 65536 context, q8_0 K/V on GPU, batch 16384 and microbatch 9216. The packaged executable and run-server.cmd accept --prefill-kv. Existing profiles and no-argument decode selection remain available.

The named preset processed the complete sample-doc.txt plus sample-prompt.txt without KV/batch/context arguments supplied separately: 8564 prompt tokens, zero cache reuse and 512 generated tokens. Prefill measured 101.75 tokens/s and decode 3.12 tokens/s. Input, formatted-prompt and output hashes matched the earlier explicit configuration's 117.68/3.12 t/s run. Diagnostics were enabled at trace verbosity in both; Windows file-cache state and background usage were uncontrolled.

Native long aliases and the negative KV option successfully overrode context, batch, microbatch, both KV types and placement, verified through actual model allocations. Profile and launcher help, conflicting-profile rejection and the existing privacy regression passed. Inputs and ggml-cuda.dll were unchanged, CUDA graphs were untouched, and both test servers were stopped. The executable was rebuilt and packaged using the existing generated compile/link commands after the normal Ninja build stalled; no build-system or performance bottleneck fixes were made.

Measured memory, exact commands, native logs, GPU samples and Windows RAM/disk/page-input/dedicated/shared-memory counters are retained in matching folders: [preset results summary](bespoke-tests/2026-10-10-prefill-kv/SUMMARY.md).
