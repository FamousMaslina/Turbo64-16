"""Local benchmark orchestration; no project profile or inference changes."""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SERVER = ROOT / "turbo64-16/bin/llama-server.exe"
MODEL = Path("C:/Users/Tudi/Documents/Tudi/AI/MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf")
URL = "http://127.0.0.1:5567"


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def request(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as response:
        return json.load(response)


def configs():
    result = []
    for batch, microbatch in [(4096, 4096), (16384, 9216)]:
        for ctx, kv in [(65536, "f16"), (65536, "q8_0"), (32768, "f16"), (32768, "q8_0")]:
            result.append({"name": f"gpu-ctx{ctx}-kv{kv}-b{batch}-ub{microbatch}", "ctx": ctx,
                           "kv": kv, "batch": batch, "microbatch": microbatch, "placement": "gpu"})
    for ctx, kv in [(65536, "f16"), (65536, "q8_0"), (32768, "f16"), (32768, "q8_0")]:
        result.append({"name": f"cpu-only-ctx{ctx}-kv{kv}-b4096-ub4096", "ctx": ctx,
                       "kv": kv, "batch": 4096, "microbatch": 4096, "placement": "cpu-only"})
    result.insert(4, {"name": "reference-cpu-kv-ctx65536-kvf16-b16384-ub9216", "ctx": 65536,
                      "kv": "f16", "batch": 16384, "microbatch": 9216, "placement": "cpu-kv"})
    return result


def prepare_benchmark():
    source = (ROOT / "turbo64-16/benchmark.py").read_text(encoding="utf-8")
    old = 'if report["server"]["default_generation_settings"]["n_ctx"] != 65536 or report["server"]["total_slots"] != 1:'
    new = 'if report["server"]["default_generation_settings"]["n_ctx"] not in (32768, 65536) or report["server"]["total_slots"] != 1:'
    if old not in source:
        raise RuntimeError("Benchmark context guard changed; inspect before running.")
    source = source.replace(old, new)
    source = source.replace("The document benchmark requires one slot with a 65536-token context.",
                            "The document benchmark requires one 32768- or 65536-token slot.")
    source = source.replace("len(document_tokens) + args.generate >= 65536",
                            'len(document_tokens) + args.generate >= report["server"]["default_generation_settings"]["n_ctx"]')
    source = source.replace("The complete document and generation must fit in the 64K context.",
                            "The complete document and generation must fit in the allocated context.")
    path = OUTPUT / "benchmark_context_copy.py"
    path.write_text(source, encoding="utf-8")
    return path


def telemetry(directory, stopped, server_pid):
    with (directory / "gpu.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc", "memory_used_mib", "memory_free_mib", "gpu_util_percent", "memory_util_percent",
                         "power_w", "temperature_c", "sm_clock_mhz", "memory_clock_mhz"])
        while not stopped.is_set():
            sampled = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            for row in csv.reader(sampled.stdout.splitlines()):
                writer.writerow([timestamp()] + [value.strip() for value in row])
            handle.flush()
            point = {"utc": timestamp(), "server_pid": server_pid}
            try:
                point["slots"] = [{key: slot.get(key) for key in ("id", "id_task", "is_processing", "n_ctx", "n_prompt_tokens_processed", "n_prompt_tokens_cache", "next_token")}
                                  for slot in request("/slots", timeout=2)]
                for slot in point["slots"]:
                    token = slot.get("next_token")
                    if isinstance(token, dict):
                        slot["next_token"] = {key: token.get(key) for key in ("n_decoded", "n_remain")}
                    elif isinstance(token, list):
                        slot["next_token"] = [{key: item.get(key) for key in ("n_decoded", "n_remain")} for item in token if isinstance(item, dict)]
            except (OSError, urllib.error.URLError, TimeoutError):
                point["slots_unavailable"] = True
            with (directory / "progress.jsonl").open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(point) + "\n")
            stopped.wait(5)


def run_config(config, benchmark, phase, repetitions, generate):
    directory = OUTPUT / config["name"] / phase
    directory.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("LLAMA_ARG_", "TURBO64_16"))}
    env.update({"TURBO64_16": "2", "TURBO64_16_DIAGNOSTICS": "1", "TURBO64_16_READAHEAD_ASYNC": "1",
                "TURBO64_16_READAHEAD_MIB": "128", "LLAMA_ARG_LOG_VERBOSITY": "4", "LLAMA_ARG_JINJA": "1"})
    command = [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "5567", "-c", str(config["ctx"]),
               "-ngl", "0" if config["placement"] == "cpu-only" else "all", "--n-cpu-moe", "47",
               "-t", "16", "-tb", "24", "-b", str(config["batch"]), "-ub", str(config["microbatch"]),
               "-fa", "on", "-lm", "mmap", "-np", "1", "--jinja", "--reasoning", "on", "--no-warmup",
               "-ctk", config["kv"], "-ctv", config["kv"], "--metrics", "--slots", "--log-verbosity", "4",
               "--log-file", str(directory / "server.technical.log"),
               "--no-kv-offload" if config["placement"] != "gpu" else "--kv-offload"]
    if config["placement"] == "cpu-only":
        command.extend(["--device", "none", "--no-op-offload"])
    metadata = {"config": config, "phase": phase, "started_utc": timestamp(), "command": command,
                "environment_overrides": {key: value for key, value in env.items() if key.startswith(("TURBO64_16", "LLAMA_ARG_"))},
                "repetitions": repetitions, "generate": generate, "diagnostics_enabled": True,
                "status": "starting"}
    write_json(directory / "config.json", metadata)
    stopped = threading.Event()
    worker = None
    counters = None
    server = None
    run = None
    with (directory / "server.stdout.log").open("w", encoding="utf-8") as stdout, (directory / "server.stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            metadata["server_pid"] = server.pid
            write_json(directory / "config.json", metadata)
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    metadata["status"] = "startup_failed"
                    metadata["server_exit_code"] = server.returncode
                    return metadata
                try:
                    request("/health")
                    break
                except (OSError, urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup exceeded five minutes.")
                    time.sleep(1)
            props = request("/props")
            if props["default_generation_settings"]["n_ctx"] != config["ctx"] or props["total_slots"] != 1:
                raise RuntimeError("Allocated context does not match the requested configuration.")
            write_json(directory / "props.json", props)
            counter_names = [r"\Memory\Available MBytes", r"\Memory\Pages Input/sec", r"\Memory\Page Reads/sec",
                             r"\PhysicalDisk(1 C:)\Disk Read Bytes/sec", r"\PhysicalDisk(1 C:)\Avg. Disk sec/Read",
                             r"\PhysicalDisk(1 C:)\Current Disk Queue Length", r"\GPU Adapter Memory(*)\Dedicated Usage",
                             r"\GPU Adapter Memory(*)\Shared Usage"]
            if config["placement"] != "cpu-only":
                counter_names.extend([f"\\GPU Process Memory(pid_{server.pid}*)\\Dedicated Usage",
                                      f"\\GPU Process Memory(pid_{server.pid}*)\\Shared Usage"])
            write_json(directory / "counter_names.json", counter_names)
            counters = subprocess.Popen(["typeperf", *counter_names, "-si", "5", "-f", "CSV", "-o",
                                         str(directory / "windows-counters.csv"), "-y"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            worker = threading.Thread(target=telemetry, args=(directory, stopped, server.pid), daemon=True)
            worker.start()
            metadata["status"] = "benchmarking"
            metadata["benchmark_started_utc"] = timestamp()
            write_json(directory / "config.json", metadata)
            args = [sys.executable, str(benchmark), "--url", URL, "--document", str(ROOT / "turbo64-16/sample-doc.txt"),
                    "--prompt", str(ROOT / "turbo64-16/sample-prompt.txt"), "--repetitions", str(repetitions),
                    "--generate", str(generate), "--label", config["name"] + "-" + phase,
                    "--output", str(directory / "results.json")]
            with (directory / "benchmark.console.log").open("w", encoding="utf-8") as console:
                run = subprocess.Popen(args, cwd=ROOT, stdout=console, stderr=console, creationflags=subprocess.CREATE_NO_WINDOW)
                metadata["benchmark_pid"] = run.pid
                write_json(directory / "config.json", metadata)
                exit_code = run.wait(timeout=14400)
            metadata["benchmark_exit_code"] = exit_code
            metadata["status"] = "complete" if exit_code == 0 else "benchmark_failed"
            if exit_code == 0:
                report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
                metadata["prefill_tokens_per_second"] = report["median_prompt_tokens_per_second"]
                metadata["decode_tokens_per_second"] = report["median_decode_tokens_per_second"]
                metadata["tokens_requested"] = report["tokens_requested"]
        except Exception as error:
            metadata["status"] = "error"
            metadata["error_type"] = type(error).__name__
        finally:
            stopped.set()
            if run is not None and run.poll() is None:
                run.terminate()
                run.wait(timeout=15)
            if counters is not None and counters.poll() is None:
                counters.terminate()
                counters.wait(timeout=15)
            if worker is not None:
                worker.join(timeout=15)
            if server is not None and server.poll() is None:
                server.terminate()
                server.wait(timeout=15)
            metadata["finished_utc"] = timestamp()
            metadata["server_stopped"] = server is None or server.poll() is not None
            write_json(directory / "config.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--phase", default="matrix-tg64")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--generate", type=int, default=64)
    args = parser.parse_args()
    try:
        request("/health", timeout=1)
    except (OSError, urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError("Benchmark port already has a live server.")
    benchmark = prepare_benchmark()
    matrix = configs()
    if args.only:
        names = set(args.only)
        matrix = [config for config in matrix if config["name"] in names]
        if len(matrix) != len(names):
            parser.error("Unknown configuration name.")
    for config in matrix:
        print(f"START {config['name']} {args.phase} {timestamp()}", flush=True)
        result = run_config(config, benchmark, args.phase, args.repetitions, args.generate)
        with (OUTPUT / "matrix-index.jsonl").open("a", encoding="utf-8") as index:
            index.write(json.dumps(result) + "\n")
        print(f"END {config['name']} {result['status']} prefill={result.get('prefill_tokens_per_second')} decode={result.get('decode_tokens_per_second')} {timestamp()}", flush=True)


if __name__ == "__main__":
    main()
