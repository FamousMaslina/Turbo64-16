# --prefill-kv results - 2026-10-10

The packaged preset and launcher passed validation. Defaults are 65536 context, GPU q8_0 K/V, batch 16384, microbatch 9216, the original MiMo model, CPU MoE 47, 16/24 threads, Flash Attention, mmap, one Jinja slot and reasoning. Existing profiles and no-argument decode selection are preserved.

One uncached request used the complete sample-doc.txt and sample-prompt.txt: 8564 prompt tokens and 512 output tokens, cache_n=0. Input, formatted-prompt and output hashes match the earlier explicit-argument configuration. Diagnostic timers were enabled at trace verbosity in both runs.

| Run | Prefill t/s | Decode t/s | Minimum sampled free dedicated VRAM MiB |
| --- | ---: | ---: | ---: |
| Earlier explicit configuration | 117.68 | 3.12 | 412 |
| Named preset validation | 101.75 | 3.12 | 163 |

The preset's prompt evaluation took 84.17 seconds. Peak total dedicated GPU use was 15886 MiB; peak process shared GPU memory was 2294 MiB. Minimum available RAM was 167 MiB. GPU global/SWA KV allocations were 765.00/958.24 MiB, both q8_0. Compute allocations were 4854.95 MiB CUDA and 1895.02 MiB CUDA_Host.

Host staging counters totaled 48.52 seconds and requested uploads totaled 102.50 GiB. These retain the earlier bottleneck evidence; requested uploads are not physical disk reads. Windows file-cache state and background usage were uncontrolled, so the prefill difference does not establish a preset performance regression. The allocated 64K context was populated only with the benchmark document and output.

CLI and launcher help, existing profile recognition, multiple-profile rejection and the existing privacy regression passed. Native long aliases overrode context to 32768, batch/microbatch to 4096, both KV types to F16 and placement to CPU; actual allocations were checked without benchmarking that override case. Both servers were stopped. The input files and ggml-cuda.dll are byte-identical to their pre-change hashes; CUDA graphs were not changed.

Matching folders retain exact arguments, properties, timings/hashes, server logs, GPU samples and Windows memory/disk/page-input counters. See [the GPU run](gpu-ctx65536-kvq8_0-b16384-ub9216/config.json), [results](gpu-ctx65536-kvq8_0-b16384-ub9216/results.json), [override check](native-alias-overrides-cpu-kv-ctx32768-kvf16-b4096-ub4096/config.json), [validation](validation.json) and [measurement data](measurement-summary.json).

The wrapper was compiled, linked and packaged successfully using the existing generated commands after the normal Ninja invocation stalled. Its build evidence is retained; no build-system or performance bottleneck fixes were made. See [run notes](README.md).
