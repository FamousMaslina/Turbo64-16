# Turbo64-16

Opt-in Windows optimization for MiMo-V2.6-Flash on Ryzen 7950X, 64 GB RAM, Gen5 NVMe, RTX 4080. Based on the supplied llama.cpp-b11475 source archive. Maintained independently of ggml-org/llama.cpp; no upstream submission is intended.

The document benchmark uses `sample-doc.txt` and `sample-prompt.txt`, formatted by the model's chat template into 8,564 tokens. Two runs of the rebuilt prefill profile measured **90.48-112.66 tokens/s**, with a **101.57 tokens/s median**, compared with the initial **15.01 tokens/s** default and **42.55 tokens/s** previous prefill profile. It retains 65,536 context capacity and F16 KV, with KV stored in CPU RAM to make room for a 9,216-token microbatch. Decode's median was 1.54 tokens/s versus the starting 1.51. Windows file-cache state was uncontrolled; see [results](RESULTS.md) for all measurements and limitations.

## Run

From the source directory in Command Prompt:

```bat
turbo64-16\run-server.cmd --turbo-prefill
```

This uses the bespoke `turbo64-16\bin\llama-server.exe` and your original model, 64K context, `--n-cpu-moe 47`, 16 decode threads, 24 batch threads, batch 16384, microbatch 9216, CPU F16 KV, Flash Attention, mmap, Jinja, and reasoning. It listens on all IPv4 interfaces (`0.0.0.0:5559`), including LAN. Stop any existing server on that port first. No-argument startup and `run-server.cmd 2` still select the original 1024 microbatch decode settings.

Select a named profile directly on the packaged executable:

```bat
turbo64-16\bin\llama-server.exe --turbo-prefill
turbo64-16\bin\llama-server.exe --turbo-prefill2
turbo64-16\bin\llama-server.exe --prefill-kv
turbo64-16\bin\llama-server.exe --turbo-decode
turbo64-16\bin\llama-server.exe --turbo-balanced
```

| Flag | Batch | Microbatch | KV placement | Purpose |
| --- | ---: | ---: | --- | --- |
| `--turbo-prefill` | 16384 | 9216 | CPU, explicit F16 | Process the complete benchmark document in one large expert-transfer pass |
| `--turbo-prefill2` | 4096 | 4096 | GPU, default F16 | Restore the prefill configuration from before the CPU-KV changes |
| `--prefill-kv` | 16384 | 9216 | GPU, explicit Q8 K/V | Use the tested large-microbatch 64K GPU KV configuration |
| `--turbo-decode` | 4096 | 1024 | GPU, default F16 | Retain the original generation profile |
| `--turbo-balanced` | 4096 | 2048 | GPU, default F16 | Retain the intermediate batch and memory footprint |

All five supply the same model, 64K context, CPU MoE 47, 16 decode/24 batch threads, Flash Attention, mmap, one slot, Jinja, reasoning, and default 128 MiB asynchronous read-ahead in mode 2. The CPU prefill profile adds `--no-kv-offload -ctk f16 -ctv f16`. The `--prefill-kv` preset adds `--kv-offload -ctk q8_0 -ctv q8_0`. These are measured configuration choices, not guarantees across workloads or file-cache states. CUDA graphs are unchanged.

Run the 64K GPU Q8 preset with `turbo64-16\run-server.cmd --prefill-kv`. Its configuration measured 117.68 tokens/s prefill and 3.12 tokens/s output in the earlier complete-document/512-token test. That run had 412 MiB minimum free dedicated VRAM, including desktop usage; see [the configuration matrix](bespoke-tests/2026-10-10-kv-context/SUMMARY.md) for memory measurements and file-cache limitations.

The rebuilt named preset passed a full-document/512-token check at 101.75 tokens/s prefill and 3.12 tokens/s output, with the same input/output hashes as that earlier run. Native context, batch, microbatch, K/V type and negative placement overrides passed. See [preset validation](bespoke-tests/2026-10-10-prefill-kv/SUMMARY.md) for logs, memory samples and build details.

Two full-document runs of the restored `--turbo-prefill2` profile measured 38.05-39.13 tokens/s prefill and 2.72-2.85 tokens/s output over 512 generated tokens, with medians of 38.59 and 2.78. Later output intervals measured about 3.1-3.3 tokens/s, while the complete output averages remained below 3. Run it with `turbo64-16\run-server.cmd --turbo-prefill2`; see RESULTS.md for the methodology and cache limitations.

Choose only one profile per launch. Native options override its defaults, including aliases and negative options. For example, `llama-server.exe --turbo-prefill --ubatch-size 6144 --kv-offload` uses the tested 6144 microbatch with GPU KV. Select `--turbo-prefill2` to reproduce the previous prefill profile directly. Existing environment overrides remain available for read-ahead and diagnostics. With no arguments the executable selects the decode profile; custom arguments without a profile retain native-argument behavior. `--help` lists the profiles.

The launcher also accepts these flags and forwards further server arguments: `turbo64-16\run-server.cmd --turbo-prefill`. The existing numeric mode and explicit microbatch syntax remains available.

## Connect your app over LAN

The Ethernet LAN address observed on this PC is `192.168.0.234`. Use this OpenAI-compatible base URL:

```text
http://192.168.0.234:5559/v1
```

Use `http://127.0.0.1:5559/v1` for an app running on the same PC. Request `GET /v1/models` to obtain the model ID. Chat uses `POST /v1/chat/completions`; streaming uses the usual `stream: true`. No API key is configured by the launcher; an SDK that requires a value can use a placeholder. The LAN IP can change if the router assigns a new address. If Windows asks for network access, allow the executable on your Private network. Firewall settings are not modified by this project.

## Bespoke diagnostics

The dedicated executable prints a TURBO ASCII banner and enables debug verbosity, timestamps, performance counters, slots monitoring, and Prometheus metrics. Every diagnostic launch saves a timestamped log under `turbo64-16/bin/logs/` while also printing diagnostics in the terminal.

Additional Turbo reports include:

- Expert tensor name, quantization format, backend, batch token count, selected/total experts, requested upload bytes, chunk count, and pinned-buffer use.
- Time in Windows prefetch calls, host memcpy staging, staging-buffer event waits, upload enqueue calls, and total host processing per expert tensor.
- Per-graph totals for expert matrices, upload bytes, chunks, the same timing categories, submission time, and return status.
- CPU usage, process working set/private commit/peak working set, system RAM availability/load, cumulative page faults, and logical process read deltas every five seconds.
- HTTP status and body byte counts, KV allocations, graph splits, placement, and final prompt/generation timings.

API monitoring is available at `http://192.168.0.234:5559/metrics`, `/slots`, and `/props`. These use llama.cpp's existing schemas. OpenAI chat responses retain the existing response format. The new detailed transfer/hardware reports are in the console/log file, not custom JSON fields.

`requested` upload bytes are not measured physical NVMe reads. Prefetch timing is time inside the OS hint, not a measurement of all disk work. Host staging time can include page-fault stalls on mapped weights. Enqueue and graph submission timings do not wait for all GPU work to complete. Page faults include soft faults; logical read deltas are not total physical disk throughput and can exclude mapped-file reads. GPU model/KV/compute allocations still appear in the existing CUDA logs; live dedicated/shared GPU usage should be checked in Task Manager or nvidia-smi.

The bespoke executable enforces metadata-only logging. Prompt/message bodies, responses, reasoning text, tool arguments, final parsed messages, token text/IDs, parser AST dumps, and arbitrary error text are excluded from console and file logs. The policy stays active with `--verbose` and `--log-jsonl` and cannot be disabled through server arguments. Structured content dumps are blocked; JSONL output can still carry technical counters. `--log-prompts-dir` and `--verbose-prompt` are rejected, and the prompt-file writer is disabled even if configured through a preset. Normal API responses still reach your app. Earlier log files are not retroactively scrubbed.

Logging many details can affect benchmark throughput. For performance measurement, disable the extra diagnostic timers and hardware reporter before launch:

```bat
set TURBO64_16_DIAGNOSTICS=0
turbo64-16\run-server.cmd --turbo-prefill
```

The default logger then uses info verbosity. Set `LLAMA_ARG_LOG_VERBOSITY=4` for trace or `5` for debug. Set `LLAMA_ARG_LOG_FILE` to choose another file, or pass `--log-file`. Other native server arguments override the bespoke host/port/logging defaults. The full MiMo profile is supplied when no arguments are given or when a named Turbo profile is selected. Otherwise include `-m` and the model settings you want.

The environment variable `TURBO64_16` selects the behavior when the process starts:

| Mode | Behavior |
| --- | --- |
| 0 or unset | Original source behavior, including normal CPU repacking and startup prefetch |
| 1 | File-backed CPU expert weights, no whole-model startup prefetch, bounded selected-expert read-ahead |
| 2 | Mode 1 plus 2 x 32 MiB host staging buffers with event-protected asynchronous GPU uploads and bounded asynchronous read-ahead |

Mode 2 groups selected ranges into windows and prepares the next window while staging the current one. At most one future window is in flight. `TURBO64_16_READAHEAD_MIB` sets the selected bytes per window (32-256, default 128); `TURBO64_16_READAHEAD_ASYNC=0` uses synchronous read-ahead for comparison. Mode 1 always uses synchronous read-ahead. All prefetch tasks finish before the tensor copy callback returns.

Asynchronous `prefetch_ms` measures worker time and can overlap `host_stage_ms`; these counters must not be added together. `prefetch_wait_ms` measures host time waiting for a read-ahead task. Prefetch completion does not prove physical disk I/O completion or page residency.

The custom upload path handles batches of at least 32 tokens. Smaller batches use the original copy callback. Expert placement stays file-backed for the process lifetime, so CPU decode also uses unrepacked experts; decode performance can change. Switching modes requires restarting the server. Host staging does not allocate additional expert caches in VRAM. Backend capability/allocation failure falls back to prefetch-only transfers.

## Measure

With the server running, in another terminal:

```bat
py turbo64-16\benchmark.py --document turbo64-16\sample-doc.txt --prompt turbo64-16\sample-prompt.txt --repetitions 3 --generate 64 --label document-prefill
```

Results are written to `turbo64-16\results-<label>.json`. The document mode reads both UTF-8 files in full, applies the model's chat template, and uses the exact same tokenized prompt in every run. It requires one 65536-token slot, disables prompt caching, and rejects any cached tokens or an incomplete prompt count. It records prefill/decode timing, wall time, server properties, token counts, and input/output hashes without saving conversation text. Allocating a 64K context with this 8564-token prompt does not measure attention at a filled 64K context.

Repeat the same workload and include both initial and later runs. Windows file caching makes a single result noisy. Do not compare a cached prompt against an uncached one. Report prompt/generation speeds, dedicated/shared GPU memory from Task Manager, RAM use, and whether normal chat output looks correct. Do not run stock and Turbo simultaneously.

The older synthetic mode remains available with `--tokens` and `--workload <name>` for transfer sanity checks. It varies prompt prefixes and rejects reused prefixes. Use the document mode for performance changes as required by AGENTS.md.

The pre-update executable and DLLs are preserved locally under `baseline-log-build/`; `run-server.cmd baseline` runs that build with the original 1024 microbatch. `run-server.cmd 2 2048` or `run-server.cmd 2 4096` chooses a microbatch explicitly. The baseline binaries are local artifacts and are not included in the source patch.

The prefill profile's 9216 microbatch amortizes expert reads over the complete benchmark document; the larger logical batch also avoids a split at 4096 tokens. CPU KV placement releases persistent GPU KV storage while retaining F16 precision and 64K capacity. Observed total dedicated GPU memory was approximately 15,500 MiB including desktop applications, with limited headroom. More desktop GPU usage or a different workload can change available headroom. Shared GPU usage was not measured in these document runs. The GPU-KV 6144 alternative also used about 15,500 MiB and had faster decode in the initial sample. See RESULTS.md for exact figures.

## Architecture and approach

The local GGUF metadata confirms 48 layers, hidden size 4096, 256 routed experts with 8 selected per token, expert width 2048, a 128-token sliding window, 39 sliding-window layers and 9 global-attention layers. The first layer has a dense FFN. The two GGUF files total 126,214,594,144 bytes (117.55 GiB); expert tensors alone total 111.625 GiB. Gate/up experts use Q2_K, down experts use MXFP4. The file name does not imply every tensor is Q2_K.

On this machine, CPU expert repacking can allocate anonymous copies of mapped weights. Retaining ordinary CPU buffers lets weights remain file-backed and lets the existing scheduler offload supported prefill matmuls to CUDA. Decode still runs through the normal scheduler. This removes an expert-weight duplication path; it does not make the entire model fit in RAM or eliminate disk reads.

The existing selected-expert copy callback supplies actual routing IDs. Turbo coalesces consecutive selected experts, preserves the original MMQ padding, prefetches bounded windows using the existing Windows helper, and uploads chunks of at most 32 MiB. Two pinned host buffers let the host prepare the next chunk while a previous transfer completes. Events prevent overwriting a buffer still in use by the GPU; destruction waits for pending transfers. Read-ahead overlaps the current window's staging and never predicts future-layer routing. There is no cross-layer GPU compute/copy pipeline, persistent GPU expert cache, model rewriting, or model-weight quantization change. CUDA graphs and CUDA kernel sources are unchanged.

Sources:

- [Xiaomi MiMo-V2.6-Flash-MOPD model architecture](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
- [Official model configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/raw/main/config.json)
- [Microsoft PrefetchVirtualMemory documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory)
- [NVIDIA transfer synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
- [Existing llama.cpp offloaded-MoE transfer discussion](https://github.com/ggml-org/llama.cpp/issues/25859)
- [Existing CPU_REPACK allocation report](https://github.com/ggml-org/llama.cpp/issues/23603)

## Build and validate

```bat
turbo64-16\build.cmd
set TURBO64_16=2
build-turbo64-16\bin\test-turbo64-16.exe
set TURBO64_16=1
build-turbo64-16\bin\test-turbo64-16.exe
build-turbo64-16\bin\test-backend-ops.exe test -b CUDA0 -o MUL_MAT_ID -p "type_a=(q2_K|mxfp4)"
build-turbo64-16\bin\test-turbo64-16-private-logs.exe turbo64-16\private-log-regression.jsonl
```

The build script uses installed MSVC 2022 Build Tools, Ninja/CMake, CUDA from PATH, Release, native CPU instructions, and GPU architecture 89. It builds server, the bespoke server, bench, and validation tools. The bespoke executable and project DLLs are copied into turbo64-16/bin/. Do not build while either server is running: Windows locks loaded DLLs. The server UI uses llama.cpp's existing asset provisioning, which may download UI assets during the first build. Model inference and benchmarks are local.

The transfer test covers exact bytes for both expert formats, dense/sparse/contiguous/final/empty expert selections, 128 MiB window boundaries, 32 MiB chunks, padding, buffer reuse, destruction with pending transfers, and the single-token fallback. The existing backend tests compare CUDA expert matmuls against CPU. Full-model perplexity and deterministic token equivalence have not been established; CPU/CUDA computation can differ numerically.

The private-log regression runs at maximum verbosity with JSONL enabled and forces chat parser debug mode. It checks that message canaries, token traces, structured payloads, and parser dumps never reach exported logs while hardware, graph, HTTP, and prompt timing counters still appear. It also verifies that parsing still returns the message to the caller.

## Update or remove

Ten existing source files have hooks: the original four under `src/`, the build hook in `tools/server/CMakeLists.txt`, logging hooks in `common/log.h` and `common/log.cpp`, the parser-debug guard in `common/chat.cpp`, and guards in `tools/server/server-context.cpp` and `tools/server/server-http.cpp`. The Turbo implementation, private logging policy, tests, launch/build/benchmark scripts, documentation, and patch bundle live in `turbo64-16/`. No ggml backend files, public llama API, CUDA graphs, CUDA kernels, or MiMo graph files are modified.

Original copies are under `turbo64-16/original/` with their source paths preserved. `source-hooks.patch` contains just the small upstream hooks; `turbo64-16.patch` also includes the new standalone files.

For a fresh llama.cpp source tree, use `git -c core.autocrlf=false apply --check <path-to-turbo64-16.patch>` and then `git -c core.autocrlf=false apply <path-to-turbo64-16.patch>` from that tree. `git apply` also works on source archives without a Git repository. Rebuild with the included script. If an update changes the callback or loader interfaces, inspect the rejected hunk instead of force-applying it. The patches are validated against this archive only.

To remove the changes from this tree, stop the server and use `git -c core.autocrlf=false apply --reverse --check turbo64-16/source-hooks.patch` followed by `git -c core.autocrlf=false apply --reverse turbo64-16/source-hooks.patch`. This removes the source hooks and leaves artifacts for inspection. Check first because reversing must not overwrite later edits. Alternatively, run mode 0 to disable the optimization without editing files.

Measured results and limitations are recorded in `RESULTS.md`.
