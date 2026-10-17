"""Run with `colab exec -s SESSION -f scripts/colab_smoke.py --timeout 1200`.

Upload dist/gasp-colab.zip to gasp-colab.zip on the runtime first. This uses
an isolated CPU environment and saves /content/gasp-smoke-results.zip.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


archive = Path("/gasp-colab.zip")
root = Path("/content/gasp")
results = Path(tempfile.mkdtemp(prefix="gasp-smoke-", dir="/content"))
started = time.perf_counter()
report = {"success": False, "commands": [], "python": sys.version,
          "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def run(label, args):
    print(f"Starting {label}", flush=True)
    start = time.perf_counter()
    with (results/f"{label}.log").open("w") as log:
        completed = subprocess.run(list(map(str, args)), cwd=root,
            env={**os.environ, "PYTHONPATH": str(root/"src"), "PYTHONUNBUFFERED": "1",
                 "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
            stdout=log, stderr=subprocess.STDOUT, text=True)
    report["commands"].append({"label": label, "args": list(map(str, args)),
        "returncode": completed.returncode, "seconds": time.perf_counter()-start})
    print((results/f"{label}.log").read_text()[-2500:], flush=True)
    completed.check_returncode()


try:
    assert (3, 11) <= sys.version_info[:2] < (3, 14), "Requires Python 3.11–3.13"
    with zipfile.ZipFile(archive) as source:
        for member in source.namelist():
            if not member.startswith("gasp/") or ".." in Path(member).parts:
                raise ValueError("Unexpected archive member")
        source.extractall("/content")
    python = Path("/content/gasp-smoke-venv/bin/python")
    # Colab's system Python may omit ensurepip. Bootstrap pip using the system
    # pip's supported --python option instead of ensurepip.
    run("venv", [sys.executable, "-m", "venv", "--without-pip", python.parent.parent])
    run("pip-bootstrap", [sys.executable, "-m", "pip", "--python", python, "install", "pip"])
    run("torch-install", [python, "-m", "pip", "install", "torch==2.7.1",
                          "--index-url", "https://download.pytorch.org/whl/cpu"])
    run("dependencies", [python, "-m", "pip", "install", "transformers==4.57.6",
        "tokenizers==0.22.2", "numpy==2.2.6", "pytest==9.0.2", "math-verify==0.8.0",
        "peft==0.18.0", "accelerate==1.12.0"])
    run("package-install", [python, "-m", "pip", "install", "--no-deps", "."])
    run("packages", [python, "-m", "pip", "freeze"])
    run("tests", [python, "-m", "pytest", "-q", "--junitxml", results/"tests.xml"])
    run("smoke", [python, "-m", "gasp", "smoke", "--output", results/"run"])
    report["success"] = True
finally:
    report["wall_seconds"] = time.perf_counter()-started
    (results/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    shutil.make_archive("/content/gasp-smoke-results", "zip", results)
    print(json.dumps(report, indent=2), flush=True)
    print("Saved /content/gasp-smoke-results.zip", flush=True)
