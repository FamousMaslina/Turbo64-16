# Turbo64-16

Opt-in Windows optimization for MiMo-V2.6-Flash on Ryzen 7950X, 64 GB RAM, Gen5 NVMe, RTX 4080. Based on the supplied llama.cpp-b11475 source archive. Maintained independently of ggml-org/llama.cpp; no upstream submission is intended.

## Run

From the source directory in Command Prompt:

```bat
turbo64-16\run-server.cmd 2
```

This uses the bespoke `turbo64-16\bin\llama-server.exe` and your original model, 64K context, `--n-cpu-moe 47`, 16 decode threads, 24 batch threads, batch 4096, microbatch 1024, Flash Attention, mmap, Jinja, and reasoning. It listens on all IPv4 interfaces (`0.0.0.0:5559`), including LAN. Stop any existing server on that port first. You can also run the bespoke executable without arguments to use the same model profile.

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
turbo64-16\run-server.cmd 2
```

The default logger then uses info verbosity. Set `LLAMA_ARG_LOG_VERBOSITY=4` for trace or `5` for debug. Set `LLAMA_ARG_LOG_FILE` to choose another file, or pass `--log-file`. Other native server arguments override the bespoke host/port/logging defaults. When passing your own arguments directly to the executable, include `-m` and the model settings you want; the full MiMo profile is supplied automatically only when no arguments are given.

The environment variable `TURBO64_16` selects the behavior when the process starts:

| Mode | Behavior |
| --- | --- |
| 0 or unset | Original source behavior, including normal CPU repacking and startup prefetch |
| 1 | File-backed CPU expert weights, no whole-model startup prefetch, bounded selected-expert read-ahead |
| 2 | Mode 1 plus 2 x 32 MiB host staging buffers with event-protected asynchronous GPU uploads |

The custom upload path handles batches of at least 32 tokens. Smaller batches use the original copy callback. Expert placement stays file-backed for the process lifetime, so CPU decode also uses unrepacked experts; decode performance can change. Switching modes requires restarting the server. Host staging does not allocate additional expert caches in VRAM. Backend capability/allocation failure falls back to prefetch-only transfers.

## Measure

With the server running, in another terminal:

```bat
py turbo64-16\benchmark.py --tokens 1024 --repetitions 3 --generate 32 --label turbo2
```

Restart with `turbo64-16\run-server.cmd 0` and repeat with `--label original`. Use mode 1 and `--label turbo1` to isolate staging's contribution. Results are written to `turbo64-16\results-<label>.json`. Each prompt has a different prefix and disables caching; the script rejects reused prefixes. It records prompt processing, generation timing, wall time, server properties, and token counts, without saving generated message text. A allocated 64K context with a 1024-token prompt does not measure attention at a filled 64K context.

Repeat the same workload and include both initial and later runs. Windows file caching makes a single result noisy. Do not compare a cached prompt against an uncached one. Report prompt/generation speeds, dedicated/shared GPU memory from Task Manager, RAM use, and whether normal chat output looks correct. Do not run stock and Turbo simultaneously.

If testing larger prompts, try changing `-ub 1024` to `-ub 2048` and then `-ub 4096` in the launcher, keeping `-b 4096`. Larger microbatches amortize expert reads across more tokens but need more compute memory. These values are experiments, not validated defaults. Stop increasing the microbatch if shared GPU memory appears or performance drops.

## Architecture and approach

The local GGUF metadata confirms 48 layers, hidden size 4096, 256 routed experts with 8 selected per token, expert width 2048, a 128-token sliding window, 39 sliding-window layers and 9 global-attention layers. The first layer has a dense FFN. The two GGUF files total 126,214,594,144 bytes (117.55 GiB); expert tensors alone total 111.625 GiB. Gate/up experts use Q2_K, down experts use MXFP4. The file name does not imply every tensor is Q2_K.

On this machine, CPU expert repacking can allocate anonymous copies of mapped weights. Retaining ordinary CPU buffers lets weights remain file-backed and lets the existing scheduler offload supported prefill matmuls to CUDA. Decode still runs through the normal scheduler. This removes an expert-weight duplication path; it does not make the entire model fit in RAM or eliminate disk reads.

The existing selected-expert copy callback supplies actual routing IDs. Turbo coalesces consecutive selected experts, preserves the original MMQ padding, prefetches at most 128 MiB per window using the existing Windows helper, and uploads chunks of at most 32 MiB. Two pinned host buffers let the host prepare the next chunk while a previous transfer completes. Events prevent overwriting a buffer still in use by the GPU; destruction waits for pending transfers. There is no speculative routing, cross-layer GPU compute/copy pipeline, persistent GPU expert cache, model rewriting, or quantization change.

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

Ten existing source files have small hooks: the original four under `src/`, the build hook in `tools/server/CMakeLists.txt`, logging hooks in `common/log.h` and `common/log.cpp`, the parser-debug guard in `common/chat.cpp`, and guards in `tools/server/server-context.cpp` and `tools/server/server-http.cpp`. The Turbo implementation, private logging policy, tests, launch/build/benchmark scripts, documentation, and patch bundle live in `turbo64-16/`. No ggml kernels, public llama API, or MiMo graph files are modified.

Original copies are under `turbo64-16/original/` with their source paths preserved. `source-hooks.patch` contains just the small upstream hooks; `turbo64-16.patch` also includes the new standalone files.

For a fresh llama.cpp source tree, use `git -c core.autocrlf=false apply --check <path-to-turbo64-16.patch>` and then `git -c core.autocrlf=false apply <path-to-turbo64-16.patch>` from that tree. `git apply` also works on source archives without a Git repository. Rebuild with the included script. If an update changes the callback or loader interfaces, inspect the rejected hunk instead of force-applying it. The patches are validated against this archive only.

To remove the changes from this tree, stop the server and use `git -c core.autocrlf=false apply --reverse --check turbo64-16/source-hooks.patch` followed by `git -c core.autocrlf=false apply --reverse turbo64-16/source-hooks.patch`. This removes the source hooks and leaves artifacts for inspection. Check first because reversing must not overwrite later edits. Alternatively, run mode 0 to disable the optimization without editing files.

Measured results and limitations are recorded in `RESULTS.md`.
