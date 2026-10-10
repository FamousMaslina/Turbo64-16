"""Validate the named preset against the real document and native CLI overrides."""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SERVER = ROOT / "turbo64-16/bin/llama-server.exe"
spec = importlib.util.spec_from_file_location("matrix", OUTPUT.parent / "2026-10-10-kv-context/run_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)
ENV = {key: value for key, value in os.environ.items() if not key.startswith(("LLAMA_ARG_", "TURBO64_16"))}
ENV.update({"TURBO64_16": "2", "TURBO64_16_DIAGNOSTICS": "1", "LLAMA_ARG_LOG_VERBOSITY": "4"})


def validate_case(name, extra, ctx, batch, microbatch, backend, kv, benchmark=False):
    directory = OUTPUT / name
    directory.mkdir(exist_ok=False)
    command = [str(SERVER), "--prefill-kv", "--host", "127.0.0.1", "--port", "5567", "--no-warmup",
               "--log-file", str(directory / "server.technical.log"), *extra]
    metadata = {"command": command, "environment_overrides": {k: v for k, v in ENV.items() if k.startswith("TURBO64_16") or k.startswith("LLAMA_ARG_")},
                "expected": {"ctx": ctx, "batch": batch, "microbatch": microbatch, "kv_backend": backend, "kv_type": kv},
                "started_utc": matrix.timestamp(), "status": "starting"}
    matrix.write_json(directory / "config.json", metadata)
    stopped = threading.Event()
    worker = counters = server = None
    with (directory / "server.stdout.log").open("w") as stdout, (directory / "server.stderr.log").open("w") as stderr:
        try:
            server = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            metadata["server_pid"] = server.pid
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {server.returncode}")
                try:
                    matrix.request("/health")
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Startup exceeded five minutes")
                    time.sleep(1)
            props = matrix.request("/props")
            matrix.write_json(directory / "props.json", props)
            assert props["default_generation_settings"]["n_ctx"] == ctx and props["total_slots"] == 1
            log = (directory / "server.technical.log").read_text(encoding="utf-8")
            for key, value in [("n_ctx", ctx), ("n_batch", batch), ("n_ubatch", microbatch)]:
                assert re.search(rf"llama_context: {key}\s+= {value}\b", log), key
            assert len(re.findall(rf"{backend}\s+KV buffer size", log)) == 2
            assert len(re.findall(rf"K \({kv}\):.*V \({kv}\):", log)) == 2
            metadata["configuration_verified"] = True
            if benchmark:
                names = [r"\Memory\Available MBytes", r"\Memory\Pages Input/sec", r"\Memory\Page Reads/sec",
                         r"\PhysicalDisk(1 C:)\Disk Read Bytes/sec", r"\PhysicalDisk(1 C:)\Avg. Disk sec/Read",
                         r"\PhysicalDisk(1 C:)\Current Disk Queue Length", r"\GPU Adapter Memory(*)\Dedicated Usage",
                         r"\GPU Adapter Memory(*)\Shared Usage", f"\\GPU Process Memory(pid_{server.pid}*)\\Dedicated Usage",
                         f"\\GPU Process Memory(pid_{server.pid}*)\\Shared Usage"]
                matrix.write_json(directory / "counter_names.json", names)
                counters = subprocess.Popen(["typeperf", *names, "-si", "5", "-f", "CSV", "-o", str(directory / "windows-counters.csv"), "-y"],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                worker = threading.Thread(target=matrix.telemetry, args=(directory, stopped, server.pid), daemon=True)
                worker.start()
                metadata["status"] = "benchmarking"
                matrix.write_json(directory / "config.json", metadata)
                print(f"BENCHMARK {name}: preset configuration verified", flush=True)
                args = [sys.executable, str(ROOT / "turbo64-16/benchmark.py"), "--url", matrix.URL,
                        "--document", str(ROOT / "turbo64-16/sample-doc.txt"), "--prompt", str(ROOT / "turbo64-16/sample-prompt.txt"),
                        "--repetitions", "1", "--generate", "512", "--label", "prefill-kv-preset", "--output", str(directory / "results.json")]
                metadata["benchmark_command"] = args
                matrix.write_json(directory / "config.json", metadata)
                with (directory / "benchmark.console.log").open("w") as console:
                    subprocess.run(args, cwd=ROOT, env=ENV, stdout=console, stderr=console, timeout=1800, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                report = json.loads((directory / "results.json").read_text())
                reference = json.loads((OUTPUT.parent / "2026-10-10-kv-context/gpu-ctx65536-kvq8_0-b16384-ub9216/confirm-tg512/results.json").read_text())
                assert report["tokens_requested"] == 8564 and report["runs"][0]["timings"]["predicted_n"] == 512
                for field in ["document_sha256", "question_sha256"]:
                    assert report[field] == reference[field], field
                for field in ["prompt_sha256", "response_sha256"]:
                    assert report["runs"][0][field] == reference["runs"][0][field], field
                metadata.update({"prefill_tokens_per_second": report["median_prompt_tokens_per_second"],
                                 "decode_tokens_per_second": report["median_decode_tokens_per_second"], "reference_hashes_match": True})
            metadata["status"] = "passed"
        except Exception as error:
            metadata.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            stopped.set()
            if counters is not None and counters.poll() is None:
                counters.terminate()
                counters.wait(timeout=15)
            if worker is not None:
                worker.join(timeout=15)
            if server is not None and server.poll() is None:
                server.terminate()
                server.wait(timeout=15)
            metadata.update({"server_stopped": server is None or server.poll() is not None, "finished_utc": matrix.timestamp()})
            matrix.write_json(directory / "config.json", metadata)
    print(f"PASS {name}: prefill={metadata.get('prefill_tokens_per_second')} decode={metadata.get('decode_tokens_per_second')}", flush=True)


def main():
    try:
        matrix.request("/health", timeout=1)
    except (OSError, TimeoutError):
        pass
    else:
        raise RuntimeError("Test port already occupied")
    with (OUTPUT / "cli.console.log").open("w") as console:
        for flag in ["--prefill-kv", "--turbo-prefill", "--turbo-prefill2", "--turbo-decode", "--turbo-balanced"]:
            run = subprocess.run([str(SERVER), flag, "--help"], env=ENV, capture_output=True, text=True, timeout=30)
            console.write(run.stdout + run.stderr)
            assert run.returncode == 0 and flag in run.stderr
        run = subprocess.run([ENV["COMSPEC"], "/c", str(ROOT / "turbo64-16/run-server.cmd"), "--prefill-kv", "--help"],
                             cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=30)
        console.write(run.stdout + run.stderr)
        assert run.returncode == 0 and "--prefill-kv" in run.stderr
        run = subprocess.run([str(SERVER), "--prefill-kv", "--turbo-prefill"], env=ENV, capture_output=True, text=True, timeout=30)
        console.write(run.stdout + run.stderr)
        assert run.returncode == 2 and "select only one" in run.stderr
        subprocess.run([str(ROOT / "build-turbo64-16/bin/test-turbo64-16-private-logs.exe"), str(OUTPUT / "privacy-regression.log")],
                       cwd=ROOT, env=ENV, stdout=console, stderr=console, timeout=30, check=True)
    validate_case("gpu-ctx65536-kvq8_0-b16384-ub9216", [], 65536, 16384, 9216, "CUDA0", "q8_0", benchmark=True)
    validate_case("native-alias-overrides-cpu-kv-ctx32768-kvf16-b4096-ub4096",
                  ["--ctx-size", "32768", "--batch-size", "4096", "--ubatch-size", "4096",
                   "--cache-type-k", "f16", "--cache-type-v", "f16", "--no-kv-offload"], 32768, 4096, 4096, "CPU", "f16")
    before = json.loads((OUTPUT / "before.json").read_text())
    unchanged = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in before.items() if not name.endswith("llama-server.exe")}
    assert all(unchanged.values())
    matrix.write_json(OUTPUT / "validation.json", {"cli_and_launcher": "passed", "conflicting_profiles_rejected": True,
                                                 "privacy_regression": "passed", "inputs_and_cuda_dll_unchanged": unchanged})


if __name__ == "__main__":
    main()
