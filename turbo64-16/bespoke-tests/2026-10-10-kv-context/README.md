# KV/context benchmark artifacts

Local tests on 2026-10-10, Europe/Bucharest. No profiles or inference code are changed.

`gpu-*` retains GPU attention/dense layers, CPU MoE 47, and GPU KV. `cpu-only-*` uses `-ngl 0 --device none --no-op-offload` and CPU KV for the entire model. `reference-cpu-kv-*` is the existing large prefill setup with CPU KV and the original GPU/model placement; it is a reference rather than a CPU-only test.

The matrix covers 32768/65536 allocated context and F16/Q8_0 keys and values. GPU configurations use both previous batch/microbatch pairs, 4096/4096 and 16384/9216. CPU-only configurations use 4096/4096. All runs use the same MiMo GGUF, 16 decode threads, 24 batch threads, Flash Attention, mmap, one slot, mode 2, and asynchronous 128 MiB read-ahead. No CUDA graph options are changed.

Every request reads the entire `sample-doc.txt` and `sample-prompt.txt`, applies the model chat template, and uses greedy sampling with seed 1234 and `cache_prompt=false`. The expected prompt is 8564 tokens with SHA256 `df4676a58139a24751179b34211264b6cad603816c8aee7503abe985bb4999a3`. Only the document and generated tokens are populated; these tests do not fill the allocated context.

The initial matrix generates 64 tokens once per configuration. Follow-up phases are named separately and record their generation length and repetitions. Diagnostics are enabled consistently in these new tests. Earlier diagnostics-disabled results are historical context rather than strictly matched comparisons. Servers run sequentially as fresh processes. Windows file caching and unrelated application load are uncontrolled.

Each configuration/phase directory contains:

- `config.json`: exact command, environment, PIDs, timestamps, status, and aggregate rates.
- `results.json`: native prefill/decode timings, counts, server properties, and input/output hashes; produced only after a completed request.
- `server.technical.log`, `server.stderr.log`, `server.stdout.log`: native allocations, transfer/graph timings, hardware observations, and errors. The technical and stderr logs overlap.
- `benchmark.console.log`: request progress and timing/hash metadata.
- `windows-counters.csv`: 5-second Windows counters for available RAM, disk reads/latency/queue, page inputs, and adapter/process dedicated/shared GPU allocations. CSV timestamps use the host's Europe/Bucharest local time.
- `gpu.csv`: timestamped NVIDIA memory, utilization, power, temperature, and clocks, sampled approximately every 5 seconds. GPU totals include desktop applications.
- `progress.jsonl`: UTC timestamps and metadata-only slot progress.
- `props.json`: allocated context and slot count from the running server.
- `counter_names.json`: the requested Windows counter paths.

`manifest.json` preserves input/binary hashes and model size/mtime. `matrix-index.jsonl` records completed attempts. `run_matrix.py` reuses an isolated copy of the existing benchmark; only that copy's context checks are widened to accept both requested capacities. The original `benchmark.py` is unchanged by this test session.

Requested expert upload bytes are not physical disk reads. Prefetch worker time can overlap host staging and must not be added to it. Graph submission time does not necessarily include completed GPU execution. Page inputs and disk counters cover the system or C: drive, not exclusively the server. GPU shared allocation does not by itself prove active device-memory eviction or spill. Sampled peaks can miss short spikes.

The source release includes summaries, exact configurations, timing/hash results and validation scripts. Detailed logs, telemetry CSVs, slot progress and startup-only artifacts remain in the local bespoke folders.
