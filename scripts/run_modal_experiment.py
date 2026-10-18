"""Run one bounded exploratory stage; GPUs stop between all stages.

Initialize once with --prepare-only; subsequent commands reuse its frozen ZIP.
This runner never generates GPT responses or waits for corpus review on a GPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import zipfile

from run_modal_benchmark import ROOT, package_source

VOLUME = "gasp-exploratory-artifacts"
STAGE_LIMITS = {"prepare": 3600, "baseline": 1800, "train": 12600}
STARTUP_ALLOWANCE = 60
EXPERIMENT_SCRIPTS = ("scripts/run_modal_experiment.py", "scripts/experiment_worker.py",
    "scripts/probe_cycle.py", "scripts/run_modal_cycle.py", "scripts/analyze_exploratory.py")


def safe_extract(archive, destination):
    if any(Path(n).is_absolute() or ".." in Path(n).parts for n in archive.namelist()):
        raise ValueError("Unsafe archive member")
    archive.extractall(destination)


def package_experiment(root, destination):
    hashes = package_source(root, destination)
    with zipfile.ZipFile(destination, "a", zipfile.ZIP_DEFLATED) as archive:
        for name in EXPERIMENT_SCRIPTS:
            archive.write(root / name, name)
            hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return hashes


def timed_commit(commit, events_path):
    """Record durable-storage latency separately from checkpoint serialization."""
    import time
    def measured():
        started = time.monotonic()
        success = False
        try:
            result = commit()
            success = True
            return result
        finally:
            event = {"event": "phase", "phase": "volume_commit", "success": success,
                     "seconds": time.monotonic() - started}
            with Path(events_path).open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
    return measured


def worker(stage, condition, run_id, attempt_id, source_hash, max_seconds, role_steps, resume,
           prior_gpu_seconds, total_gpu_seconds, volume_name):
    import contextlib
    import os
    import subprocess
    import sys
    import time
    import traceback
    import modal
    started = time.monotonic()
    root = Path("/work/gasp")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("/source.zip") as source:
        safe_extract(source, root)
    persistent = Path("/persist") / run_id
    persistent.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    attempt = persistent / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    commit = timed_commit(volume.commit, attempt / "volume-commits.jsonl")
    ledger_path = persistent / "gpu-budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "prior_gpu_seconds": prior_gpu_seconds, "total_gpu_seconds": total_gpu_seconds,
        "source_sha256": source_hash, "attempts": []}
    if (ledger["source_sha256"], ledger["prior_gpu_seconds"], ledger["total_gpu_seconds"]) != (
            source_hash, prior_gpu_seconds, total_gpu_seconds):
        raise ValueError("Experiment source or GPU budget differs from its durable ledger")
    reserved = max_seconds + STARTUP_ALLOWANCE
    used = prior_gpu_seconds + sum(r["charged_seconds"] for r in ledger["attempts"])
    if used + reserved > total_gpu_seconds:
        raise ValueError("Stage reservation exceeds remaining total GPU-hour budget")
    record = {"attempt": attempt_id, "stage": stage, "condition": condition,
              "status": "reserved", "charged_seconds": reserved,
              "hard_timeout_seconds": max_seconds, "startup_allowance_seconds": STARTUP_ALLOWANCE}
    ledger["attempts"].append(record)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    commit()
    # Open/close the log for each write: Volume.commit may reload the mount.
    class Tee:
        def __init__(self, stream):
            self.stream = stream
        def write(self, value):
            self.stream.write(value)
            with (attempt / "execution.log").open("a") as f:
                f.write(value)
            return len(value)
        def flush(self):
            self.stream.flush()
    report = {"success": False, "stage": stage, "condition": condition,
              "run_id": run_id, "attempt_id": attempt_id, "source_sha256": source_hash,
              "durable_volume": volume_name, "durable_path": str(persistent)}
    previous_cwd = os.getcwd()
    try:
        os.chdir(root)
        os.environ.update({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        sys.path[:0] = [str(root / "src"), str(root / "scripts")]
        with contextlib.redirect_stdout(Tee(sys.stdout)), contextlib.redirect_stderr(Tee(sys.stderr)):
            subprocess.run(["nvidia-smi"], check=True, timeout=30)
            if stage == "prepare" and not (persistent / "cuda-tests.json").exists():
                tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root,
                    env={**os.environ, "PYTHONPATH": str(root / "src")},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
                print(tests.stdout, flush=True)
                (persistent / "cuda-tests.json").write_text(json.dumps({
                    "returncode": tests.returncode, "output": tests.stdout}))
                tests.check_returncode()
                commit()
            inputs = root / "stage-inputs"
            if Path("/inputs.zip").exists():
                with zipfile.ZipFile("/inputs.zip") as archive:
                    safe_extract(archive, inputs)
            from experiment_worker import run
            report["result"] = run(stage, condition, persistent, inputs,
                max_seconds - (time.monotonic() - started) - 60, commit,
                role_steps=role_steps, resume=resume)
            report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        os.chdir(previous_cwd)
        report["worker_seconds"] = time.monotonic() - started
        record.update(status="finished", success=report["success"],
                      worker_seconds=report["worker_seconds"],
                      charged_seconds=report["worker_seconds"] + STARTUP_ALLOWANCE)
        ledger_path.write_text(json.dumps(ledger, indent=2))
        (attempt / "worker.json").write_text(json.dumps(report, indent=2))
        commit()
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in persistent.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                archive.write(path, path.relative_to(persistent))
    return bundle.getvalue()


def build_inputs(directory, corpus, provenance, audit):
    if not all((corpus, provenance, audit)):
        raise ValueError("Baseline needs --corpus, --provenance and --audit before provisioning")
    values = [(corpus, "held-out-corruptions.jsonl"),
              (provenance, "held-out-corruptions.jsonl.provenance.json"),
              (audit, "corpus-review.json")]
    review = json.loads(Path(audit).read_text())
    digest = hashlib.sha256(Path(corpus).read_bytes()).hexdigest()
    if (review.get("corpus_sha256") != digest or review.get("verdict") != "pass"
            or not review.get("scope") or not review.get("sample_ids")):
        raise ValueError("The semantic review must pass and match the frozen corpus")
    archive = directory / "baseline-inputs.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as saved:
        for path, name in values:
            saved.write(path, name)
    return archive


def verify_launch_sources(root, archive):
    # Runtime stage code comes from the ZIP. These files execute before extraction
    # or construct the image and must also match that frozen snapshot.
    with zipfile.ZipFile(archive) as source:
        for name in ("scripts/run_modal_experiment.py", "scripts/run_modal_benchmark.py",
                     "requirements-colab.txt", "data/full-v4/resolved_config.json"):
            if source.read(name) != (root / name).read_bytes():
                raise ValueError("Launch source differs from the frozen experiment: " + name)


def local_spend(directory, manifest):
    ledger_path = directory / "results/gpu-budget.json"
    entries = json.loads(ledger_path.read_text())["attempts"] if ledger_path.exists() else []
    used = manifest["prior_gpu_seconds"] + sum(r["charged_seconds"] for r in entries)
    known = {r["attempt"] for r in entries}
    # If a worker dies before returning artifacts, conservatively charge its full
    # reservation until the durable ledger is retrieved. Never infer zero spend.
    for path in directory.glob("*.launch.json"):
        launch = json.loads(path.read_text())
        if launch["attempt_id"] not in known:
            used += launch["max_seconds"] + STARTUP_ALLOWANCE
    return used


def claim_local_lock(directory):
    import fcntl
    handle = (directory / "launch.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another stage is running for this experiment") from None
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stage", choices=STAGE_LIMITS)
    parser.add_argument("--condition", choices=("unified", "split"))
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--role-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--prior-gpu-seconds", type=float, default=2880)
    parser.add_argument("--total-gpu-hours", type=float, default=10)
    args = parser.parse_args()
    if args.role_steps < 5 or args.role_steps % 5:
        parser.error("--role-steps must be a positive multiple of five")
    if not args.prepare_only and not args.stage:
        parser.error("Select --stage or --prepare-only")
    if args.stage == "train" and not args.condition:
        parser.error("Training requires --condition")
    if args.stage != "train" and args.condition:
        parser.error("--condition applies only to training")
    directory = args.directory or ROOT / "artifacts" / (
        "gasp-experiment-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.zip"
    manifest_path = directory / "experiment.json"
    if not archive.exists():
        hashes = package_experiment(ROOT, archive)
        (directory / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
        manifest = {"run_id": directory.name, "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "volume": VOLUME, "prior_gpu_seconds": args.prior_gpu_seconds,
            "total_gpu_seconds": args.total_gpu_hours * 3600, "gpu": "A100-80GB",
            "note": "Budget includes prior benchmark GPU time and a 60-second startup allowance per worker."}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Frozen source archive changed")
    print("Experiment: " + str(directory), flush=True)
    if args.prepare_only:
        return
    # Held until this CLI exits; kernel releases it even after interruption.
    launch_lock = claim_local_lock(directory)
    verify_launch_sources(ROOT, archive)
    limit = args.max_seconds or STAGE_LIMITS[args.stage]
    if limit < 300:
        raise ValueError("Stage must allow at least 300 seconds")
    input_archive = build_inputs(directory, args.corpus, args.provenance, args.audit) if args.stage == "baseline" else None
    used = local_spend(directory, manifest)
    if used + limit + STARTUP_ALLOWANCE > manifest["total_gpu_seconds"]:
        raise ValueError("Stage reservation exceeds the remaining total GPU budget; reduce --max-seconds")
    import modal
    cfg = json.loads((ROOT / "data/full-v4/resolved_config.json").read_text())
    download = ("from huggingface_hub import snapshot_download; "
        f"snapshot_download({cfg['model']!r}, revision={cfg['revision']!r}, "
        "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])")
    image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential")
        .pip_install_from_requirements(str(ROOT / "requirements-colab.txt"))
        .pip_install("vllm==0.10.1.1")
        .env({"HF_HOME": "/opt/hf-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
        .run_commands("python -c " + shlex.quote(download))
        .add_local_file(ROOT / "scripts/run_modal_benchmark.py", "/root/run_modal_benchmark.py")
        .add_local_file(Path(__file__), "/root/run_modal_experiment.py")
        .add_local_file(archive, "/source.zip"))
    if input_archive:
        image = image.add_local_file(input_archive, "/inputs.zip")
    volume = modal.Volume.from_name(manifest["volume"], create_if_missing=True, version=2)
    app = modal.App("gasp-experiment-" + args.stage + ("-" + args.condition if args.condition else ""))
    remote = app.function(image=image, gpu="A100-80GB", cpu=4, memory=(65536, 131072),
        volumes={"/persist": volume}, timeout=limit, startup_timeout=900, retries=0,
        min_containers=0, max_containers=1, scaledown_window=2,
        single_use_containers=True, include_source=False)(worker)
    attempt_id = args.stage + ("-" + args.condition if args.condition else "") + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    launch = {"attempt_id": attempt_id, "stage": args.stage, "condition": args.condition,
              "max_seconds": limit, "role_steps": args.role_steps, "resume": args.resume}
    (directory / (attempt_id + ".launch.json")).write_text(json.dumps(launch, indent=2))
    with modal.enable_output(), app.run():
        data = remote.remote(args.stage, args.condition, manifest["run_id"], attempt_id,
            manifest["source_sha256"], limit, args.role_steps, args.resume,
            manifest["prior_gpu_seconds"], manifest["total_gpu_seconds"], manifest["volume"])
        (directory / (attempt_id + ".zip")).write_bytes(data)
        with zipfile.ZipFile(io.BytesIO(data)) as result:
            safe_extract(result, directory / "results")
        report = json.loads((directory / "results/attempts" / attempt_id / "worker.json").read_text())
        print(json.dumps(report, indent=2), flush=True)
        if not report["success"]:
            raise RuntimeError("Stage failed; inspect " + str(directory / "results/attempts" / attempt_id))
    launch_lock.close()


if __name__ == "__main__":
    main()
