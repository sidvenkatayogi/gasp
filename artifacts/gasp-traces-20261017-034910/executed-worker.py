"""GPU worker invoked by run_colab_traces.py; no API key is uploaded.

Runs synchronously so the local owner can download artifacts and stop the GPU
in a finally block. A server-side timer also releases it if the owner disappears.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import zipfile

from google.colab import runtime

root = Path("/content/gasp")
results = root / "artifacts/traces"
results.mkdir(parents=True, exist_ok=True)
report = {"success": False, "commands": [], "started_at": time.time()}


def stop_after(seconds):
    timer = threading.Timer(seconds, runtime.unassign)
    timer.daemon = True
    timer.start()
    return timer


hard_stop = stop_after(10 * 3600)
python = Path("/content/gasp-cuda-venv/bin/python")
env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"}


def run(label, args, timeout=900, required=True):
    print(f"Starting {label}", flush=True)
    started = time.perf_counter()
    with (results / f"{label}.log").open("w") as log:
        completed = subprocess.run(list(map(str, args)), cwd=root, env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    report["commands"].append({"label": label, "returncode": completed.returncode,
                               "seconds": time.perf_counter()-started})
    print((results / f"{label}.log").read_text()[-3000:], flush=True)
    if required:
        completed.check_returncode()


try:
    for archive, destination, prefix in ((Path("/gasp-colab.zip"), Path("/content"), "gasp/"),
                                         (Path("/gasp-inputs.zip"), root, "data/")):
        report[archive.name+"_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as source:
            for member in source.namelist():
                if not member.startswith(prefix) or ".." in Path(member).parts:
                    raise ValueError("Unexpected archive member")
            source.extractall(destination)
    run("hardware", ["nvidia-smi"])
    run("venv", [sys.executable, "-m", "venv", "--without-pip", python.parent.parent])
    run("pip-bootstrap", [sys.executable, "-m", "pip", "--python", python, "install", "pip"])
    run("dependencies", [python, "-m", "pip", "install", "-r", "requirements-colab.txt"])
    run("packages", [python, "-m", "pip", "freeze"])
    resuming = (root / "data/full-v3/eval-contexts.jsonl.progress.pt").exists()
    if not resuming:
        run("tests", [python, "-m", "pytest", "-q", "--junitxml", results / "tests.xml"])
        # A failed training-memory probe must not block the independent trace job.
        run("memory-probe", [python, "scripts/probe_optimizer_memory.py"], required=False)
    else:
        print("Resuming validated source; skipping repeated tests and memory probe", flush=True)
    args = [python, "-m", "gasp", "prepare-eval-contexts",
            "--config", "data/full-v3/resolved_config.json", "--source", "data/raw/gsm_raw.jsonl",
            "--output", "data/full-v3/eval-contexts.jsonl", "--max-seconds", str(8*3600)]
    if resuming:
        args.append("--resume")
    run("contexts", args, timeout=8*3600+900)
    report["success"] = True
    report["complete"] = (root / "data/full-v3/eval-contexts.jsonl.manifest.json").exists()
except BaseException:
    report["error"] = traceback.format_exc()
    raise
finally:
    report["finished_at"] = time.time()
    (results / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        with zipfile.ZipFile("/content/gasp-traces-results.zip", "w", zipfile.ZIP_DEFLATED) as out:
            for directory in (results, root / "data/full-v3"):
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and not path.name.endswith(".tmp"):
                        out.write(path, path.relative_to(root))
    finally:
        hard_stop.cancel()
        stop_after(300)  # Download grace period, even if the local owner is lost.
    print(json.dumps(report, indent=2), flush=True)
    print("Saved /content/gasp-traces-results.zip; automatic shutdown in 5 minutes", flush=True)
