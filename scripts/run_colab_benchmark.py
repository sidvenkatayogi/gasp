"""Own exactly one benchmark runtime; never start a corpus or campaign."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from run_colab_traces import colab, gpu_session


ROOT = Path(__file__).resolve().parents[1]


def main():
    name = "gasp-vllm-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = ROOT / "artifacts" / name
    directory.mkdir()
    archive = directory / "source.zip"
    files = [p for folder in ("src", "tests", "configs") for p in (ROOT / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    files += [ROOT / p for p in ("scripts/probe_vllm.py", "requirements-colab.txt",
                                "pyproject.toml", "data/full-v4/resolved_config.json")]
    files += list((ROOT / "data/raw").glob("*.jsonl*"))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as saved:
        for path in files:
            saved.write(path, path.relative_to(ROOT))
        # Existing lifecycle tests import this script without executing main.
        saved.write(ROOT / "scripts/run_colab_traces.py", "scripts/run_colab_traces.py")
    (directory / "source-sha256.json").write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)+"\n")
    worker = directory / "executed-worker.py"
    shutil.copy2(ROOT / "scripts/colab_benchmark.py", worker)
    print("Benchmark session: " + name, flush=True)
    process = None
    with gpu_session(name):
        print(colab("upload", "-s", name, archive, "gasp-benchmark.zip").stdout, flush=True)
        try:
            with (directory / "execution.log").open("w") as log:
                process = subprocess.Popen(["colab", "--auth=oauth2", "exec", "-s", name,
                    "-f", str(worker), "--timeout", "3300"], stdout=log, stderr=subprocess.STDOUT)
                started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic()-started > 3400:
                        raise TimeoutError("Benchmark owner deadline reached")
                    time.sleep(20)
                    colab("download", "-s", name,
                          "content/gasp-benchmark/artifacts/vllm-probe/worker.json",
                          directory / "worker-live.json", check=False, timeout=20)
                    print((directory / "execution.log").read_text()[-700:], flush=True)
        finally:
            result = colab("download", "-s", name, "content/gasp-benchmark-results.zip",
                           directory / "results.zip", check=False, timeout=60)
            print(result.stdout or result.stderr, flush=True)
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    if (directory / "results.zip").exists():
        with zipfile.ZipFile(directory / "results.zip") as saved:
            if any(Path(n).name != n for n in saved.namelist()):
                raise ValueError("Unexpected result archive member")
            saved.extractall(directory / "results")
        print((directory / "results/worker.json").read_text(), flush=True)
    else:
        raise RuntimeError("No benchmark bundle downloaded; inspect logs")


if __name__ == "__main__":
    main()
