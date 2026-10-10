# 2026-10-10: KV/context configuration tests

Completed 18 requests across 12 requested variants and one existing CPU-KV reference: 13 initial requests with 64 generated tokens, plus five fresh-process confirmations with 512 generated tokens. All tests and artifacts are local. No server code, launcher, profiles, CUDA kernels, or CUDA graph settings were changed. The original benchmark.py was left untouched; an isolated copy generalized only its context validation for 32K/64K.

## Workload and controls

Windows 11, Ryzen 7950X, 64 GB DDR5, RTX 4080, NVIDIA driver 610.74. Same two local MiMo GGUF shards, totaling 117.55 GiB. Each request used the entire turbo64-16/sample-doc.txt followed by turbo64-16/sample-prompt.txt in one user message, with the model chat template, greedy sampling, seed 1234, and cache_prompt=false. Every request processed exactly 8564 prompt tokens with cache_n=0 and prompt SHA256 df4676a58139a24751179b34211264b6cad603816c8aee7503abe985bb4999a3.

All cases used one slot, 16 decode/24 batch threads, Flash Attention, mmap, reasoning, mode 2, asynchronous 128 MiB read-ahead, and no startup warmup. Diagnostics and verbosity 4 were enabled consistently, including the refreshed baselines. GPU cases retained the previous GPU/model placement and CPU MoE 47; CPU-only cases used -ngl 0 --device none --no-op-offload and CPU KV. CPU MoE 47 was retained in CPU-only commands as well. Q8 means q8_0 for both keys and values.

Only 8564 prompt tokens plus 64 or 512 generated tokens were populated; 32K/64K are allocated capacities, not filled-context performance tests. Servers ran sequentially as fresh processes. Windows file cache, background applications, and desktop GPU usage were not controlled. GPU memory numbers are sampled about every five seconds, include desktop applications, and can miss short spikes. NVIDIA reserved memory means used + free does not necessarily equal the nominal device total.

## Initial GPU matrix: 64 generated tokens

| Configuration | Batch / microbatch | Prefill t/s | Decode t/s | Peak GPU used MiB | Min GPU free MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| [GPU KV, 64K F16](gpu-ctx65536-kvf16-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 39.73 | 1.43 | 14337 | 1712 |
| [GPU KV, 64K Q8](gpu-ctx65536-kvq8_0-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 41.57 | 1.76 | 13265 | 2784 |
| [GPU KV, 32K F16](gpu-ctx32768-kvf16-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 38.00 | 1.61 | 13359 | 2690 |
| [GPU KV, 32K Q8](gpu-ctx32768-kvq8_0-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 39.75 | 1.87 | 12851 | 3198 |
| [GPU KV, 64K F16](gpu-ctx65536-kvf16-b16384-ub9216/matrix-tg64/config.json) | 16384 / 9216 | 81.80 | 1.38 | 15717 | 332 |
| [GPU KV, 64K Q8](gpu-ctx65536-kvq8_0-b16384-ub9216/matrix-tg64/config.json) | 16384 / 9216 | 124.53 | 1.96 | 15658 | 391 |
| [GPU KV, 32K F16](gpu-ctx32768-kvf16-b16384-ub9216/matrix-tg64/config.json) | 16384 / 9216 | 109.23 | 1.79 | 15731 | 318 |
| [GPU KV, 32K Q8](gpu-ctx32768-kvq8_0-b16384-ub9216/matrix-tg64/config.json) | 16384 / 9216 | 108.36 | 1.89 | 14684 | 1365 |

## Entire model on CPU: 64 generated tokens

| Configuration | Batch / microbatch | Prefill t/s | Decode t/s | Peak GPU used MiB | Min GPU free MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| [CPU-only, 64K F16](cpu-only-ctx65536-kvf16-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 21.73 | 1.37 | 850 | 15199 |
| [CPU-only, 64K Q8](cpu-only-ctx65536-kvq8_0-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 16.85 | 1.41 | 848 | 15201 |
| [CPU-only, 32K F16](cpu-only-ctx32768-kvf16-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 22.28 | 1.34 | 848 | 15201 |
| [CPU-only, 32K Q8](cpu-only-ctx32768-kvq8_0-b4096-ub4096/matrix-tg64/config.json) | 4096 / 4096 | 17.44 | 1.39 | 844 | 15205 |

CPU-only logs confirmed 0/49 GPU layers and CPU model/KV/compute buffers, with no CUDA compute buffers. The mapped model exceeds system RAM and still needs disk reads. CPU_REPACK allocated 2432.00 MiB in each CPU-only case; it is recorded as an existing allocation, not a fix made during these tests.

## Existing CPU-KV reference

The existing GPU/model placement with CPU F16 KV, 64K capacity, batch 16384/microbatch 9216 measured 107.35 t/s prefill and 1.53 t/s decode over 64 generated tokens. This is not CPU-only execution. Peak sampled total GPU use was 15569 MiB; minimum free GPU memory was 480 MiB. [Exact configuration and logs](reference-cpu-kv-ctx65536-kvf16-b16384-ub9216/matrix-tg64/config.json).

## Confirmations: 512 generated tokens

| Configuration | Batch / microbatch | Prefill t/s | Decode t/s | Peak GPU used MiB | Min GPU free MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| [GPU KV, 64K F16](gpu-ctx65536-kvf16-b4096-ub4096/confirm-tg512/config.json) | 4096 / 4096 | 40.16 | 2.74 | 13896 | 2153 |
| [GPU KV, 64K Q8](gpu-ctx65536-kvq8_0-b4096-ub4096/confirm-tg512/config.json) | 4096 / 4096 | 40.88 | 2.76 | 12832 | 3217 |
| [GPU KV, 64K Q8](gpu-ctx65536-kvq8_0-b16384-ub9216/confirm-tg512/config.json) | 16384 / 9216 | 117.68 | 3.12 | 15637 | 412 |
| [GPU KV, 32K Q8](gpu-ctx32768-kvq8_0-b16384-ub9216/confirm-tg512/config.json) | 16384 / 9216 | 118.85 | 3.03 | 14681 | 1368 |
| [CPU-only, 32K F16](cpu-only-ctx32768-kvf16-b4096-ub4096/confirm-tg512/config.json) | 4096 / 4096 | 20.08 | 1.91 | 845 | 15204 |

At the original 4096/4096 batch, 64K Q8 provided 1064 MiB more sampled minimum free VRAM than F16. Prefill and decode remained approximately unchanged: 40.88/2.76 t/s versus 40.16/2.74. This is a headroom result rather than an established speedup.

With batch 16384/microbatch 9216, the Q8 GPU cases measured 117.68 t/s at 64K and 118.85 t/s at 32K, approximately 2.93x and 2.96x the fresh 4096 F16 baseline in these observations. Complete decode averages were 3.12 and 3.03 t/s. The 32K Q8 case retained 1.34 GiB sampled minimum free VRAM, compared with 0.40 GiB for 64K Q8 at the same batch.

The selected configurations had two prefill observations, shown separately above. Material variation remains: 32K Q8 at microbatch 9216 measured 108.36 then 118.85 t/s, and CPU-only 32K F16 measured 22.28 then 20.08 t/s. The 64-token decode averages must not be compared directly with the 512-token averages. All reported decode rates are complete native averages, not selected faster intervals.

The fresh F16 baseline reproduced the earlier 512-token response hash 6a860c627aabf4affb52b1196f2e5ff6826cbb9c15a565f4fe3e561edc46c9d1. The two large-batch Q8 GPU confirmations produced the same 512-token response hash, 3c8afad96cea558f56a34ad844ef8ee259069437cbf1768d90c6aae366ac0354. Other precision/batch/placement outputs differed, so decode comparisons can include different generated content and expert routing. Answer quality, perplexity, and token equivalence across all variants were not evaluated.

## Memory reservations

Native GPU KV totals include both global and sliding-window buffers. Reducing context halves the global portion; it does not halve the sliding-window reservation. Increasing microbatch from 4096 to 9216 increases the F16 sliding-window reservation from 828.75 to 1803.75 MiB.

| Capacity / KV | KV MiB, microbatch 4096 | KV MiB, microbatch 9216 | CUDA compute MiB, microbatch 4096 | CUDA compute MiB, microbatch 9216 |
| --- | ---: | ---: | ---: | ---: |
| 64K F16 | 2268.75 | 3243.75 | 2722.17 | 4854.89 |
| 64K Q8 | 1205.27 | 1723.24 | 2722.23 | 4854.95 |
| 32K F16 | 1548.75 | 2523.75 | 2466.17 | 4278.89 |
| 32K Q8 | 822.77 | 1340.74 | 2466.23 | 4278.95 |

## Bottleneck evidence retained for later

| 512-token case | Requested staged uploads GiB | Host staging seconds | Host staging / native prefill | Peak process shared GPU allocation MiB |
| --- | ---: | ---: | ---: | ---: |
| GPU KV, 64K F16, microbatch 4096 | 276.58 | 175.13 | 82.1% | 946 |
| GPU KV, 64K Q8, microbatch 4096 | 277.10 | 172.27 | 82.2% | 946 |
| GPU KV, 64K Q8, microbatch 9216 | 102.50 | 56.39 | 77.5% | 2038 |
| GPU KV, 32K Q8, microbatch 9216 | 102.50 | 56.74 | 78.7% | 1462 |

- Measured host staging accounts for roughly 77-82% of prefill in these confirmed GPU cases. The 4096 workload uses staged passes of 4096, 372, and 4092 tokens; the larger workload uses one 8560-token staged pass. Both have a separate four-token checkpoint tail. Requested uploads fall from about 277 GiB to 102.5 GiB. The transfer path and disk-backed weights are the main measured prefill costs to investigate.
- All cases reached 99% system RAM load. Windows page-input and physical C: disk-read counters record continued reads; a near-zero process logical-read delta does not rule out mmap traffic. Whole-phase mean C: read rates ranged about 611-1364 MiB/s. These are drive/system measurements and include unrelated work. Counter means cover the whole phase, not exclusively prefill or decode; use progress.jsonl to align the phases.
- At microbatch 9216, the initial 64K F16 GPU case showed 3566 MiB peak process shared GPU allocation and near-full dedicated VRAM. Q8 reduced shared allocation and improved observed timings, but 64K Q8 still had little sampled free dedicated VRAM. Shared allocation also includes host buffers and does not by itself prove active device-memory spill.
- On CPU-only, initial Q8 prefill was 16.85-17.44 t/s versus 21.73-22.28 t/s with F16, about 22% lower throughput. Short decode averages were close. The longer CPU F16 run measured 20.08 t/s prefill and 1.91 t/s decode while retaining about 14.85 GiB free VRAM. Logs preserve CPU load, page inputs, disk latency/queue, working set, private commit, and the CPU repack allocation; no CPU bottleneck fixes were attempted.
- Prefetch worker time overlaps host staging and must not be added to it. Requested uploads are not physical NVMe reads. Graph submission timings do not necessarily include completed GPU execution. These logs identify costs and correlations; they do not isolate every CPU operator or prove a particular memory-eviction cause.

## Files and validation

[measurement-summary.csv](measurement-summary.csv) and [measurement-summary.json](measurement-summary.json) contain the aggregate measurements. [manifest.json](manifest.json) records input/binary hashes, model sizes/mtimes, and the test plan. [validation.json](validation.json) records the final checks. [README.md](README.md) describes each raw log and counter file. Each linked config.json gives the exact command and points to its matching phase directory.

All 18 requests passed prompt count, hash, cache, context, generated-count, log-presence, and CPU-placement checks. Input files, binaries, and the original benchmark.py retained their starting SHA256 hashes; model sizes/mtimes were unchanged. The local harness passed syntax, ASCII, and whitespace checks. The optional ty type checker was unavailable locally. All test servers and counter processes were stopped, and benchmark port 5567 was closed.

One startup-only orchestration attempt is preserved under startup-aborted-harness-update. It was interrupted before a benchmark request while applying explicit CPU device selection; it is excluded from the 18 completed requests and every result table.

Profiles remain unchanged. No fixes, commits, pushes, or publication were performed.
