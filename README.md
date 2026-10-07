# Turbo64-16

Windows prefill optimization and a LAN API server for MiMo-V2.6-Flash on Ryzen 7950X, 64 GB RAM, NVMe, and RTX 4080. Based on llama.cpp-b11475, with isolated patches and metadata-only server logs.

## Important Note!
This project was fully vibe coded and built entirely to serve my own needs and hardware setup. I decided to put it here in case anyone else wants to try it too on their system and for easier sharing/upkeep. Nothing else or anything serious.

See [Turbo64-16 build, launch, API, and update instructions](turbo64-16/README.md) and [validation results](turbo64-16/RESULTS.md).

## V0.2 update

- Added `--turbo-prefill`, `--turbo-decode` and `--turbo-balanced` profiles with native argument overrides.
- Grouped selected expert reads and overlapped bounded read-ahead with host staging.
- Kept CUDA graphs unchanged. This release contains source only; build instructions are linked above.

## Observed performance

Windows 11, Ryzen 7950X, 64 GB RAM, RTX 4080, MiMo-V2.6-Flash-MOPD, configured **65,536-token context**. The user repeated the same **7,482-token prompt workload** with the V0.2 prefill profile.

| Metric | Previous Turbo run | V0.2 prefill profile |
| --- | ---: | ---: |
| Prefill | 14.19 tokens/s | **47.64 tokens/s** |
| Prompt processing | 527.21 seconds | **157.06 seconds** |
| Generation | 3.18 tokens/s | **3.11 tokens/s** |
| Requested expert uploads | 665.00 GiB | **195.16 GiB** |

Prefill was **3.36x faster (+236%)**, taking **70.2% less time** and saving about **6 minutes 10 seconds**. Generation was roughly unchanged; the runs generated 663 and 421 tokens respectively, so their generation durations and total request times are not directly comparable.

This is one observed before/after comparison, without resetting the Windows file cache. It is not a universal speedup claim. The 65K figure is allocated context capacity; the prompt filled about 7.5K tokens. Requested uploads are not measured physical disk reads.

Correctness checks include **14 byte-exact Q2_K/MXFP4 transfer cases per tested configuration**, **77/77 selected CUDA-vs-CPU expert matmul checks**, and model checks for all three profiles. Methodology, earlier synthetic results and limitations are in [turbo64-16/RESULTS.md](turbo64-16/RESULTS.md).

After building, select a profile on the packaged server:

```bat
turbo64-16\bin\llama-server.exe --turbo-prefill
```

Performance varies with hardware, quantization, file-cache state, context length and batch settings. Decode optimization remains a work in progress.

Upstream licensing and credits are retained in [LICENSE](LICENSE) and [AUTHORS](AUTHORS).
