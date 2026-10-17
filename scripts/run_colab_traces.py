"""Own the A100 lifecycle; download results and stop before any OpenAI calls.

Run under caffeinate on macOS so the local cleanup/download owner stays awake.
The remote worker additionally enforces a hard deadline and an idle shutdown.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def colab(*args, check=True, timeout=180):
    command = ["colab", "--auth=oauth2", *map(str, args)]
    try:
        return subprocess.run(command, cwd=ROOT, check=check, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise
        # Monitoring is best effort: a slow Contents API must not kill GPU work.
        message = f"Colab {args[0]} timed out after {timeout}s; continuing."
        print(message, flush=True)
        return subprocess.CompletedProcess(command, 124, stdout="", stderr=message)


def backup_contexts(session, destination):
    destination.mkdir(exist_ok=True)
    # Download checkpoint first, then an audit at least as recent as it.
    for suffix in ("progress.pt", "audit.jsonl"):
        name = "eval-contexts.jsonl."+suffix
        colab("download", "-s", session, "content/gasp/data/full-v4/"+name,
              destination / name, check=False, timeout=30)


@contextmanager
def gpu_session(name):
    try:
        print(colab("new", "-s", name, "--gpu", "A100").stdout, flush=True)
        yield
    finally:
        result = colab("stop", "-s", name, check=False)
        print(result.stdout, flush=True)
        if result.returncode:
            print("STOP NEEDS CHECK: " + result.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="gasp-traces-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--generate-corpus", action="store_true")
    args = parser.parse_args()
    result_dir = ROOT / "artifacts" / args.session
    result_dir.mkdir(parents=True, exist_ok=False)
    source = ROOT / "dist/gasp-colab.zip"
    shutil.copy2(ROOT / "dist/manifest.json", result_dir / "source-manifest.json")
    inputs = result_dir / "inputs.zip"
    files = list((ROOT / "data/raw").glob("*.jsonl*"))
    files.append(ROOT / "data/full-v4/resolved_config.json")
    files += list((ROOT / "data/full-v4").glob("eval-contexts.jsonl.*"))
    if (ROOT / "data/full-v4/eval-contexts.jsonl").exists():
        raise FileExistsError("Completed contexts already exist; no GPU is needed")
    with zipfile.ZipFile(inputs, "w", zipfile.ZIP_DEFLATED) as out:
        for path in files:
            if not path.name.endswith(".tmp"):
                out.write(path, path.relative_to(ROOT))
    # Persist exact worker text as well as the archive's source hashes.
    worker = result_dir / "executed-worker.py"
    shutil.copy2(ROOT / "scripts/colab_prepare.py", worker)
    with gpu_session(args.session):
        print(colab("upload", "-s", args.session, source, "gasp-colab.zip").stdout, flush=True)
        print(colab("upload", "-s", args.session, inputs, "gasp-inputs.zip").stdout, flush=True)
        process = None
        try:
            with (result_dir / "execution.log").open("w") as log:
                process = subprocess.Popen(["colab", "--auth=oauth2", "exec", "-s", args.session,
                    "-f", str(worker), "--timeout", str(9*3600)], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT)
                polls = 0
                while process.poll() is None:
                    time.sleep(30)
                    polls += 1
                    if polls % 2 == 0:
                        live = colab("sessions", check=False, timeout=30)
                        if live.returncode == 0 and args.session not in live.stdout:
                            process.terminate()
                            process.wait(timeout=30)
                            raise RuntimeError("Colab runtime ended; inspect periodic backups before resuming")
                    # Contents API transfers do not interrupt the working kernel.
                    stage = "contexts" if polls > 6 else "dependencies"
                    colab("download", "-s", args.session,
                        f"content/gasp/artifacts/traces/{stage}.log", result_dir / f"{stage}.log",
                        check=False, timeout=30)
                    if polls % 2 == 0:
                        backup_contexts(args.session, result_dir / f"backup-{polls:05d}")
                    if (result_dir / "contexts.log").exists():
                        lines = (result_dir / "contexts.log").read_text().splitlines()
                        progress = [line for line in lines if line.startswith("contexts:")]
                        if progress:
                            print(progress[-1], flush=True)
        finally:
            # Always try to preserve the report, including after remote errors.
            download = colab("download", "-s", args.session, "content/gasp-traces-results.zip",
                             result_dir / "results.zip", check=False)
            print(download.stdout or download.stderr, flush=True)
            if download.returncode:
                backup_contexts(args.session, result_dir / "backup-final")
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    # gpu_session has released the runtime before validation/API generation.
    with zipfile.ZipFile(result_dir / "results.zip") as saved:
        for member in saved.namelist():
            if ".." in Path(member).parts or not member.startswith(("data/full-v4/", "artifacts/traces/")):
                raise ValueError("Unexpected result archive member")
        saved.extractall(result_dir / "downloaded")
    downloaded = result_dir / "downloaded"
    report = json.loads((downloaded / "artifacts/traces/report.json").read_text())
    for path in (downloaded / "data/full-v4").iterdir():
        shutil.copy2(path, ROOT / "data/full-v4" / path.name)
    print(json.dumps(report, indent=2), flush=True)
    if not report["success"]:
        raise RuntimeError("Remote preparation failed; inspect the saved report before resuming")
    if not report["complete"]:
        print("Time budget reached; GPU stopped and resumable progress downloaded.", flush=True)
        return False
    sys.path.insert(0, str(ROOT / "src"))
    from gasp.data import read_cache
    read_cache(ROOT / "data/full-v4/eval-contexts.jsonl", "eval_contexts")
    if args.generate_corpus:
        # Read only the API key; never upload it, print it or execute .env text.
        if not os.environ.get("OPENAI_API_KEY"):
            for line in (ROOT / ".env").read_text().splitlines():
                line = line.strip().removeprefix("export ").lstrip()
                name, separator, value = line.partition("=")
                if separator and name.strip() == "OPENAI_API_KEY":
                    values = shlex.split(value, comments=True)
                    if len(values) == 1:
                        os.environ["OPENAI_API_KEY"] = values[0]
        from gasp.eval_corpus import generate_eval_corpus
        result = generate_eval_corpus(ROOT / "data/full-v4/eval-contexts.jsonl",
                                     ROOT / "data/full-v4/held-out-corruptions.jsonl")
        (result_dir / "corpus-result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    return True


if __name__ == "__main__":
    while not main():
        # Each bounded chunk releases its GPU before packaging the next one.
        # Stop on any error; only clean time-budget boundaries continue.
        print("Continuing the remaining questions in a fresh bounded session.", flush=True)
