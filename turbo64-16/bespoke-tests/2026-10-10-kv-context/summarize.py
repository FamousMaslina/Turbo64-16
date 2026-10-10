"""Summarize native timing and sampled bottleneck evidence from local runs."""

import csv
import json
import re
import statistics
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def summarize(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    row = {"config": config["config"]["name"], "phase": config["phase"], "status": config["status"],
           "generate": config["generate"], "repetitions": config["repetitions"]}
    result_path = directory / "results.json"
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        row.update({"prefill_tps": report["median_prompt_tokens_per_second"],
                    "decode_tps": report["median_decode_tokens_per_second"],
                    "prompt_tokens": report["tokens_requested"],
                    "prompt_hashes": sorted({run["prompt_sha256"] for run in report["runs"]}),
                    "response_hashes": sorted({run["response_sha256"] for run in report["runs"]}),
                    "cache_counts": [run["timings"].get("cache_n", 0) for run in report["runs"]],
                    "generated_counts": [run["timings"]["predicted_n"] for run in report["runs"]],
                    "prompt_seconds": [run["timings"]["prompt_ms"] / 1000 for run in report["runs"]]})
    gpu_path = directory / "gpu.csv"
    if gpu_path.exists():
        with gpu_path.open(encoding="utf-8", newline="") as handle:
            samples = list(csv.DictReader(handle))
        for key in ("memory_used_mib", "memory_free_mib", "gpu_util_percent", "memory_util_percent", "power_w"):
            values = [value for sample in samples if (value := number(sample.get(key))) is not None]
            if values:
                row[key + "_max"] = max(values)
                row[key + "_min"] = min(values)
                row[key + "_median"] = statistics.median(values)
    counter_path = directory / "windows-counters.csv"
    if counter_path.exists():
        with counter_path.open(encoding="utf-8", newline="") as handle:
            samples = list(csv.DictReader(handle))
        columns = list(samples[0]) if samples else []
        groups = {
            "ram_free_mib": [key for key in columns if key and key.endswith("\\Available MBytes")],
            "page_inputs_per_sec": [key for key in columns if key and key.endswith("\\Pages Input/sec")],
            "disk_read_bytes_per_sec": [key for key in columns if key and key.endswith("\\Disk Read Bytes/sec")],
            "disk_read_latency_seconds": [key for key in columns if key and key.endswith("\\Avg. Disk sec/Read")],
            "disk_queue": [key for key in columns if key and key.endswith("\\Current Disk Queue Length")],
            "process_gpu_dedicated_bytes": [key for key in columns if key and "GPU Process Memory" in key and key.endswith("\\Dedicated Usage")],
            "process_gpu_shared_bytes": [key for key in columns if key and "GPU Process Memory" in key and key.endswith("\\Shared Usage")],
            "adapter_gpu_shared_bytes": [key for key in columns if key and "GPU Adapter Memory" in key and key.endswith("\\Shared Usage")],
        }
        for label, keys in groups.items():
            values = []
            for sample in samples:
                parts = [value for key in keys if (value := number(sample.get(key))) is not None]
                if parts:
                    values.append(sum(parts))
            if values:
                row[label + "_min"] = min(values)
                row[label + "_max"] = max(values)
                row[label + "_median"] = statistics.median(values)
                row[label + "_mean"] = statistics.mean(values)
    log_path = directory / "server.stderr.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        row["allocations"] = [line for line in log.splitlines() if any(marker in line for marker in
                              ("model buffer size", "KV buffer size", "compute buffer size", "offloaded", "flash_attn", "n_batch ", "n_ubatch ", "n_ctx "))]
        graph_totals = {}
        for line in log.splitlines():
            if "Turbo64-16 graph:" not in line:
                continue
            fields = {key: float(value) for key, value in re.findall(r"(\w+)=([0-9.]+)", line)}
            if fields.get("upload_requested", 0) <= 0:
                continue
            for key, value in fields.items():
                graph_totals[key] = graph_totals.get(key, 0) + value
        row["staged_prefill_graph_totals"] = graph_totals
        hardware = [dict((key, float(value)) for key, value in re.findall(r"(\w+)=([0-9.]+)", line))
                    for line in log.splitlines() if "Turbo64-16 hardware:" in line]
        for key in ("cpu", "working_set", "private_commit", "peak_working_set", "ram_load", "page_faults"):
            values = [sample[key] for sample in hardware if key in sample]
            if values:
                row[key + "_max"] = max(values)
                row[key + "_median"] = statistics.median(values)
        row["error_lines"] = [line for line in log.splitlines() if any(marker in line.lower() for marker in
                              ("failed", "out of memory", "cuda error", "not supported", "error:"))][-15:]
        row["expert_token_batches"] = sorted({int(value) for value in re.findall(r"Turbo64-16 expert:.*? tokens=(\d+)", log)})
    return row


def main():
    rows = [summarize(path.parent) for path in sorted(OUTPUT.glob("*/*/config.json"))]
    (OUTPUT / "measurement-summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["config", "phase", "status", "generate", "repetitions", "prefill_tps", "decode_tps", "memory_used_mib_max", "memory_free_mib_min",
              "process_gpu_dedicated_bytes_max", "process_gpu_shared_bytes_max", "ram_free_mib_min", "cpu_median",
              "disk_read_bytes_per_sec_mean", "disk_read_bytes_per_sec_max", "disk_read_latency_seconds_mean",
              "page_inputs_per_sec_mean", "working_set_max", "private_commit_max"]
    with (OUTPUT / "measurement-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        prefill = row.get("prefill_tps")
        decode = row.get("decode_tps")
        rates = f"{prefill:.2f} / {decode:.2f}" if prefill is not None and decode is not None else row["status"]
        print(f"{row['config']} {row['phase']}: {rates}")


if __name__ == "__main__":
    main()
