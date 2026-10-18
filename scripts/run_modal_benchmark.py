"""Bounded Modal hardware comparison; no corpus generation or training campaign.

Usage: .venv/bin/python scripts/run_modal_benchmark.py --gpus A100-80GB H100!
The source upload is an explicit allowlist. No .env or API credentials are sent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "artifacts/gasp-vllm-20261017-053328/results/vllm.json"
GPU_CHOICES = ("A100-40GB", "A100-80GB", "H100!", "H200")


def package_source(root, destination):
    files = [p for folder in ("src", "tests", "configs")
             for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".txt", ".json")
             and not any(part.startswith(".") for part in p.relative_to(root).parts)
             and not any(part.endswith((".egg-info", ".dist-info")) for part in p.parts)
             and "__pycache__" not in p.parts]
    files += [root / p for p in ("scripts/probe_vllm.py", "scripts/run_colab_traces.py",
              "scripts/run_modal_benchmark.py", "pyproject.toml", "requirements-colab.txt",
              "data/full-v4/resolved_config.json", "data/raw/gsm_raw.jsonl",
              "data/raw/gsm_raw.jsonl.manifest.json", "data/raw/math_raw.jsonl",
              "data/raw/math_raw.jsonl.manifest.json", FIXTURE)]
    hashes = {}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = str(path.relative_to(root))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, name)
    return hashes


def worker(gpu, fixture):
    # Keep remote dependencies explicit; the allowlisted runner module is shipped
    # as source, avoiding Python 3.13 -> 3.12 function-pickle incompatibilities.
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    import zipfile

    root = Path("/work/gasp")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("/source.zip") as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe source archive member")
        archive.extractall(root)
    out = root / "artifacts/vllm-probe"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1"}
    report = {"success": False, "requested_gpu": gpu, "commands": [],
              "started_at": time.time(), "fixed_backward_fixture": fixture,
              "fixture_sha256": hashlib.sha256((root / fixture).read_bytes()).hexdigest()}
    deadline = time.monotonic() + 25*60

    def run(label, args, timeout):
        print("Starting " + label, flush=True)
        started = time.perf_counter()
        with (out / (label + ".log")).open("w") as log:
            process = subprocess.run(args, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT,
                                     timeout=max(1, min(timeout, deadline-time.monotonic())))
        report["commands"].append({"label": label, "returncode": process.returncode,
                                   "seconds": time.perf_counter()-started})
        print((out / (label + ".log")).read_text()[-1800:], flush=True)
        process.check_returncode()

    try:
        run("hardware", ["nvidia-smi"], 30)
        run("packages", [sys.executable, "-m", "pip", "freeze"], 30)
        run("tests", [sys.executable, "-m", "pytest", "-q"], 180)
        for phase, timeout in (("hf", 240), ("inference", 600)):
            run(phase, [sys.executable, "scripts/probe_vllm.py", phase], timeout)
        # Save native inference results, then replay identical trajectories for
        # hardware timing; this is a component test, not on-policy training.
        shutil.copy2(out / "vllm.json", out / "vllm-generated.json")
        shutil.copy2(root / fixture, out / "vllm.json")
        for condition in ("unified", "split"):
            label = "backward-" + condition
            run(label, [sys.executable, "scripts/probe_vllm.py", "backward",
                        "--condition", condition, "--output", label], 600)
        report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["finished_at"] = time.time()
        (out / "worker.json").write_text(json.dumps(report, indent=2)+"\n")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.iterdir():
            if path.is_file():
                archive.write(path, path.name)
    return bundle.getvalue()


def unpack_results(data, destination):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if any(Path(name).name != name for name in archive.namelist()):
            raise ValueError("Unexpected result archive member")
        archive.extractall(destination)
    report = json.loads((destination / "worker.json").read_text())
    if not report["success"]:
        raise RuntimeError("Modal benchmark failed; inspect " + str(destination / "worker.json"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", choices=GPU_CHOICES, default=["A100-80GB", "H100!"])
    parser.add_argument("--prepare-only", action="store_true", help="Package inputs without contacting Modal")
    args = parser.parse_args()
    directory = ROOT / "artifacts" / ("gasp-modal-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    directory.mkdir()
    archive = directory / "source.zip"
    hashes = package_source(ROOT, archive)
    (directory / "source-sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print("Results: " + str(directory), flush=True)
    if args.prepare_only:
        return

    import modal
    cfg = json.loads((ROOT / "data/full-v4/resolved_config.json").read_text())
    download = ("from huggingface_hub import snapshot_download; "
                f"snapshot_download({cfg['model']!r}, revision={cfg['revision']!r}, "
                "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("build-essential")
             .pip_install_from_requirements(str(ROOT / "requirements-colab.txt"))
             .pip_install("vllm==0.10.1.1")
             .env({"HF_HOME": "/opt/hf-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
             .run_commands("python -c " + shlex.quote(download))
             .add_local_file(Path(__file__), "/root/run_modal_benchmark.py")
             .add_local_file(archive, "/source.zip"))
    # One app per GPU ensures the previous app is stopped before starting the
    # next. single_use_containers also releases the GPU immediately on return.
    for gpu in args.gpus:
        label = gpu.replace("!", "")
        app = modal.App("gasp-benchmark-" + label.lower())
        benchmark = app.function(image=image, gpu=gpu, cpu=4,
            memory=(65536, 98304), timeout=1800, startup_timeout=900, retries=0,
            min_containers=0, max_containers=1, scaledown_window=2,
            single_use_containers=True, include_source=False)(worker)
        target = directory / label
        target.mkdir()
        with modal.enable_output(), app.run():
            data = benchmark.remote(gpu, FIXTURE)
            (target / "results.zip").write_bytes(data)
            report = unpack_results(data, target / "results")
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
