# Turbo64-16

Windows prefill optimization and a LAN API server for MiMo-V2.6-Flash on Ryzen 7950X, 64 GB RAM, NVMe, and RTX 4080. Based on llama.cpp-b11475, with isolated patches and metadata-only server logs.

This project was vibe coded and built mostly to serve my own needs and hardware setup. I decided to put it here in case anyone else wants to try it too.

See [Turbo64-16 build, launch, API, and update instructions](turbo64-16/README.md) and [validation results](turbo64-16/RESULTS.md).

## Observed performance

These results come from my Windows 11 setup running MiMo-V2.6-Flash with a configured **65,536-token context**.

### Longer prompt

A **7,482-token prompt** completed prompt processing in about **527 seconds**:

| Run | Prefill | Generation |
| --- | ---: | ---: |
| Previous stock reference | 8.35 tokens/s | 3.83 tokens/s |
| Turbo64-16 | **14.19 tokens/s** | **3.18 tokens/s** |
| Difference from the reference | **About 1.70x / +70%** | **About 17% lower** |

The Turbo run generated **663 tokens** at an average of 3.18 tokens/s. The current optimization favors **prompt ingestion over token generation**; improving decode performance remains a work in progress.

### Shorter uncached prompts

The 1,024-token tests reached:

- **20.56 tokens/s** prefill on the first run.
- **18.44 tokens/s** prefill on the repeat run.

These rates are approximately **2.21-2.46x** the previously reported 8.35 tokens/s stock prefill figure.

### How to interpret these results

**This is not yet a controlled stock-vs-Turbo A/B benchmark.** The stock reference comes from an earlier run; the Turbo tests may differ in binary, cache state, and other runtime conditions. Treat these as **observed results on my machine, not a universal speedup claim**. A configured 65K context also does not mean the tests filled that entire context.

Correctness checks include byte-exact expert-transfer validation for **Q2_K and MXFP4** and **77/77 selected CUDA-vs-CPU expert matrix multiplication checks passing**. Configuration, methodology, local raw-result references, limitations, and correctness details are in [turbo64-16/RESULTS.md](turbo64-16/RESULTS.md).

Performance will vary with model quantization, RAM capacity and speed, storage, GPU, CPU, cache state, context length, and batch settings.

Upstream licensing and credits are retained in [LICENSE](LICENSE) and [AUTHORS](AUTHORS).
