"""Measure uncached prefill on a running llama-server using only Python's standard library."""

import argparse
import datetime
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5559")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--label", default="turbo2")
    parser.add_argument("--workload", help="Use a repeatable prompt prefix across fresh server launches for matched A/B runs.")
    parser.add_argument("--document", type=Path, help="Prefill this entire UTF-8 document instead of synthetic text.")
    parser.add_argument("--prompt", type=Path, help="UTF-8 question to append to the document using the model's chat template.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens < 32 or args.tokens + args.generate >= 65536 or args.generate < 1 or args.repetitions < 1:
        parser.error("Use 32 <= tokens, 1 <= generate, tokens + generate < 65536, repetitions >= 1.")
    if bool(args.document) != bool(args.prompt):
        parser.error("Use --document and --prompt together.")

    request(args.url, "/health")
    report = {
        "label": args.label,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "server": request(args.url, "/props"),
        "tokens_requested": args.tokens,
        "runs": [],
    }
    document_tokens = None
    if args.document:
        if report["server"]["default_generation_settings"]["n_ctx"] != 65536 or report["server"]["total_slots"] != 1:
            parser.error("The document benchmark requires one slot with a 65536-token context.")
        document = args.document.read_text(encoding="utf-8-sig")
        question = args.prompt.read_text(encoding="utf-8-sig")
        formatted = request(args.url, "/apply-template", {
            "messages": [{"role": "user", "content": document + "\n\n" + question}],
        })["prompt"]
        document_tokens = request(args.url, "/tokenize", {"content": formatted, "parse_special": True})["tokens"]
        if not document_tokens or len(document_tokens) + args.generate >= 65536:
            parser.error("The complete document and generation must fit in the 64K context.")
        report["tokens_requested"] = len(document_tokens)
        report["document_sha256"] = hashlib.sha256(args.document.read_bytes()).hexdigest()
        report["question_sha256"] = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
        report["workload"] = "document-chat-template"
    prose = "Explain how memory capacity, disk bandwidth, and transfer latency affect large language model inference. "
    for index in range(args.repetitions):
        if document_tokens is not None:
            tokens = document_tokens
        else:
            # Change the start of every synthetic prompt to prevent prefix reuse.
            prefix = (f"Independent benchmark {args.workload} {index}. " if args.workload else
                      f"Independent benchmark {args.label} {index} {time.time_ns()}. ")
            tokens = request(args.url, "/tokenize", {"content": prefix + prose * args.tokens})["tokens"][:args.tokens]
            if len(tokens) != args.tokens:
                raise RuntimeError("Tokenizer returned too few tokens.")
        print(f"Run {index + 1}/{args.repetitions}: {len(tokens)} prompt tokens, {args.generate} generated tokens.", flush=True)
        started = time.perf_counter()
        result = request(args.url, "/completion", {
            "prompt": tokens,
            "n_predict": args.generate,
            "temperature": 0,
            "seed": 1234,
            "cache_prompt": False,
            "stream": False,
        })
        if result["timings"].get("cache_n", 0) != 0 or result["timings"]["prompt_n"] != len(tokens):
            raise RuntimeError("The server reused a prompt prefix or processed a different token count; discard this run.")
        run = {"wall_seconds": time.perf_counter() - started, "timings": result["timings"],
               "tokens_cached": result.get("tokens_cached"),
               "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               "response_sha256": hashlib.sha256(result.get("content", "").encode()).hexdigest()}
        report["runs"].append(run)
        report["median_prompt_tokens_per_second"] = statistics.median(item["timings"]["prompt_per_second"] for item in report["runs"])
        report["median_decode_tokens_per_second"] = statistics.median(item["timings"]["predicted_per_second"] for item in report["runs"])
        print(json.dumps(run), flush=True)
        output = args.output or Path(__file__).parent / f"results-{args.label}.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Median prefill: {report['median_prompt_tokens_per_second']:.2f} tokens/s. Saved {output}")


if __name__ == "__main__":
    main()
