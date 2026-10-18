"""Bounded paired cycle timing on Modal; no API calls or full training campaign."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import zipfile

from run_modal_benchmark import ROOT, package_source


def prepare_inputs():
    import torch
    fixture = ROOT / "artifacts/gasp-modal-20261017-061436/A100-80GB/results/vllm-generated.json"
    saved = json.loads(fixture.read_text())
    train = [{k: v for k, v in c["row"].items() if k != "reference_trace"}
             for c in saved["contexts"]]
    response_path = ROOT / "data/full-v3/held-out-corruptions.jsonl.responses/000002.json"
    saved_response = json.loads(response_path.read_text())
    progress = torch.load(ROOT / "data/full-v3/eval-contexts.jsonl.progress.pt",
                          map_location="cpu", weights_only=False)
    context = next(r for r in progress["retained"] if r["id"] == saved_response["context_id"])
    request_text = saved_response["request"]["input"][1]["content"]
    if context["prefix"] not in request_text or context["clean"] not in request_text:
        raise ValueError("Saved API response does not match the saved context")
    visible = "".join(c["text"] for item in saved_response["response"]["output"]
                      if item["type"] == "message" for c in item["content"]
                      if c["type"] == "output_text")
    from gasp.verification import pollution
    polluted, tagged = pollution(visible)
    if not tagged:
        raise ValueError("Existing API diagnostic lacks a valid pollution block")
    model = saved_response["response"]["model"]
    return {"train": train, "training_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "eval": context | {"polluted": polluted, "generator": model},
            "eval_provenance": {"kind": "eval", "held_out": True,
                "protocol": "one-existing-GPT-corruption-timing-only",
                "generator_model": model, "generator_revision": model,
                "source_url": "https://arxiv.org/abs/2602.00173",
                "generation_settings": {k: v for k, v in saved_response["request"].items() if k != "input"},
                "subset_selection": "Existing paid preflight; runtime diagnostic only",
                "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest()}}


def worker(condition):
    import io
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    import traceback
    import zipfile
    root = Path("/work/gasp")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("/source.zip") as source:
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in source.namelist()):
            raise ValueError("Unsafe source archive")
        source.extractall(root)
    out = root / "cycle-results"
    out.mkdir()
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    report = {"success": False, "condition": condition}
    start = time.perf_counter()
    try:
        subprocess.run(["nvidia-smi"], check=True, timeout=30)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=env,
                       check=True, timeout=180)
        command = [sys.executable, "scripts/probe_cycle.py", condition,
                   "--output", str(out / condition)]
        with (out / "execution.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            # The outer Modal timeout is the hard watchdog; each line is streamed
            # so progress remains visible during long real rollout calls.
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError("Cycle subprocess failed")
        report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    report["seconds"] = time.perf_counter() - start
    (out / "worker.json").write_text(json.dumps(report, indent=2))
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in out.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".jsonl", ".log"):
                archive.write(p, p.relative_to(out))
    return bundle.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=("split", "unified"), default=["split", "unified"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "artifacts" / ("gasp-cycle-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    directory.mkdir()
    archive = directory / "source.zip"
    hashes = package_source(ROOT, archive)
    inputs = json.dumps(prepare_inputs(), indent=2).encode()
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as source:
        source.writestr("cycle-inputs.json", inputs)
        hashes["cycle-inputs.json"] = hashlib.sha256(inputs).hexdigest()
        for name in ("scripts/run_modal_cycle.py", "scripts/probe_cycle.py"):
            source.write(ROOT / name, name)
            hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (directory / "source-sha256.json").write_text(json.dumps(hashes, indent=2))
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
             .add_local_file(ROOT / "scripts/run_modal_benchmark.py", "/root/run_modal_benchmark.py")
             .add_local_file(Path(__file__), "/root/run_modal_cycle.py")
             .add_local_file(archive, "/source.zip"))
    for condition in args.conditions:
        app = modal.App("gasp-cycle-" + condition)
        benchmark = app.function(image=image, gpu="A100-80GB", cpu=4,
            memory=(65536, 131072), timeout=7200, startup_timeout=900, retries=0,
            min_containers=0, max_containers=1, scaledown_window=2,
            single_use_containers=True, include_source=False)(worker)
        target = directory / condition
        target.mkdir()
        with modal.enable_output(), app.run():
            data = benchmark.remote(condition)
            (target / "results.zip").write_bytes(data)
            with zipfile.ZipFile(io.BytesIO(data)) as result:
                if any(Path(n).is_absolute() or ".." in Path(n).parts for n in result.namelist()):
                    raise ValueError("Unsafe result archive")
                result.extractall(target)
            report = json.loads((target / "worker.json").read_text())
            print(json.dumps(report, indent=2), flush=True)
            if not report["success"]:
                raise RuntimeError("Cycle failed; inspect " + str(target))


if __name__ == "__main__":
    main()
