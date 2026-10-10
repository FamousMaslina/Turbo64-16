# --prefill-kv validation

The preset adds 65536 context, GPU Q8 K/V, batch 16384 and microbatch 9216 to the existing MiMo configuration. The benchmark launches the named preset without supplying context, batch, microbatch or KV options, then verifies their actual native allocations. It uses the complete sample-doc.txt plus sample-prompt.txt, one uncached request and 512 generated tokens. Diagnostics and trace verbosity match the earlier configuration matrix. Windows file-cache state and background activity are uncontrolled.

The GPU configuration folder contains the exact command and environment, server properties, benchmark results and hashes, console/technical logs, NVIDIA samples, Windows RAM/disk/page-input/dedicated/shared-memory counters, and token-count progress. The native-alias folder checks context, batch, microbatch, both KV types and negative KV placement overrides through actual model initialization; it does not contain a performance benchmark. cli.console.log records profile/launcher help, conflict rejection and the existing privacy regression.

Both test servers bind to loopback port 5567 and are stopped by validate.py. The script reuses telemetry from the earlier matrix. Repeating it requires a new output folder because existing case folders are preserved.

The normal build stalled inside Ninja before compiling, with no compiler child process visible; build.log and ninja-dry-run.log retain the evidence. rebuild-wrapper.cmd executes the existing generated Ninja commands for the wrapper object, executable link and package step directly. That build succeeded; rebuild-wrapper.log contains its output. No build-system or performance bottleneck fixes were made. before.json and validation.json compare benchmark inputs and ggml-cuda.dll before and after.

See SUMMARY.md for the final measurements and checks.

The source release includes summaries, exact configurations, timing/hash results and the validation script. Detailed logs, telemetry CSVs, slot progress and the local build/patch-check artifacts remain in this PC's bespoke folder.
