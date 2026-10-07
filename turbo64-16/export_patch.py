"""Export the private source patch without requiring a Git repository."""

import difflib
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    bundle = root / "turbo64-16"
    hooks = ["src/CMakeLists.txt", "src/llama-context.h", "src/llama-context.cpp", "src/llama-model-loader.cpp", "tools/server/CMakeLists.txt"]
    hooks += ["common/log.h", "common/log.cpp", "common/chat.cpp", "tools/server/server-context.cpp", "tools/server/server-http.cpp"]
    additions = [
        "expert-stream.h", "expert-stream.cpp", "test-expert-stream.cpp",
        "build.cmd", "run-server.cmd", "benchmark.py", "export_patch.py", "README.md", "RESULTS.md", "CHANGELOG.md",
        "CMakeLists.txt", "server-main.cpp", "package-server.cmake",
        "private-logs.cpp", "private-logs.h", "test-private-logs.cpp",
    ]
    hook_diff = []
    manifest = {}
    for name in hooks:
        original = bundle / "original" / name
        current = root / name
        hook_diff.extend(difflib.unified_diff(
            original.read_text(encoding="utf-8").splitlines(keepends=True),
            current.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
        manifest[name] = {
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "patched_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        }
    (bundle / "source-hooks.patch").write_text("".join(hook_diff), encoding="utf-8", newline="\n")
    complete = list(hook_diff)
    for name in additions:
        complete.extend(difflib.unified_diff(
            [], (bundle / name).read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="/dev/null", tofile="b/turbo64-16/" + name,
        ))
    (bundle / "turbo64-16.patch").write_text("".join(complete), encoding="utf-8", newline="\n")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Exported source-hooks.patch, turbo64-16.patch, and manifest.json")


if __name__ == "__main__":
    main()
