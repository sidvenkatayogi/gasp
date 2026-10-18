"""One bounded A100 diagnostic, with remote shutdown protection."""
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


def shutdown_after(seconds):
    timer = threading.Timer(seconds, runtime.unassign)
    timer.daemon = True
    timer.start()
    return timer


hard_stop = shutdown_after(55*60)
root = Path("/content/gasp-benchmark")
out = root / "artifacts/vllm-probe"
out.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1"}
report = {"success": False, "commands": [], "started_at": time.time()}
python = Path("/content/gasp-benchmark-venv/bin/python")


def run(label, args, timeout):
    print("Starting " + label, flush=True)
    start = time.perf_counter()
    with (out / (label + ".log")).open("w") as log:
        result = subprocess.run(list(map(str, args)), cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    report["commands"].append({"label": label, "returncode": result.returncode,
                               "seconds": time.perf_counter()-start})
    (out / "worker.json").write_text(json.dumps(report, indent=2)+"\n")
    print((out / (label + ".log")).read_text()[-2200:], flush=True)
    result.check_returncode()


try:
    with zipfile.ZipFile("/gasp-benchmark.zip") as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe archive member")
        archive.extractall(root)
    run("hardware", ["nvidia-smi"], 30)
    run("uv", [sys.executable, "-m", "pip", "install", "uv==0.9.2"], 90)
    run("venv", ["uv", "venv", "--python", "3.12", python.parent.parent], 180)
    run("dependencies", ["uv", "pip", "install", "--python", python,
                         "-r", "requirements-colab.txt", "vllm==0.10.1.1"], 900)
    run("packages", ["uv", "pip", "freeze", "--python", python], 30)
    run("tests", [python, "-m", "pytest", "-q"], 180)
    for phase, timeout in (("hf", 240), ("inference", 900), ("backward", 900)):
        run(phase, [python, "scripts/probe_vllm.py", phase], timeout)
    report["success"] = True
except BaseException:
    report["error"] = traceback.format_exc()
    raise
finally:
    report["finished_at"] = time.time()
    (out / "worker.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        with zipfile.ZipFile("/content/gasp-benchmark-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in out.iterdir():
                if path.is_file():
                    archive.write(path, path.name)
    finally:
        hard_stop.cancel()
        shutdown_after(180)
    print(json.dumps(report), flush=True)
