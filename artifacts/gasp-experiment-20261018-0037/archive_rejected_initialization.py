"""Preserve an inspected GPU-identity rejection before a fresh, unchanged retry."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from run_modal_experiment import claim_local_lock

condition, attempt = sys.argv[1:]
assert condition in ("unified", "split")
assert attempt.startswith("train-" + condition + "-") and "/" not in attempt
lock = claim_local_lock(HERE)
manifest = json.loads((HERE / "experiment.json").read_text())
report = json.loads((HERE / "results/attempts" / attempt / "worker.json").read_text())
assert report["success"] is False and report["condition"] == condition
assert report["error"].splitlines()[-1] == "ValueError: Shared baseline identity mismatch: gpu"
ledger_bytes = (HERE / "results/gpu-budget.json").read_bytes()
record = next(a for a in json.loads(ledger_bytes)["attempts"] if a["attempt"] == attempt)
assert record["status"] == "finished" and record["success"] is False
source = HERE / "results/train" / condition
expected = {"run.json", "environment.json"}
assert {p.name for p in source.iterdir()} == expected, "Refusing to move training output"
archive = HERE / "results/rejected-initializations" / attempt
archive.mkdir(parents=True, exist_ok=False)
v = modal.Volume.from_name(manifest["volume"])
remote_root = manifest["run_id"]
assert b"".join(v.read_file(remote_root + "/gpu-budget.json")) == ledger_bytes
remote_source = remote_root + "/train/" + condition
assert {e.path.lstrip("/") for e in v.iterdir(remote_source)} == {
    remote_source + "/" + name for name in expected
}, "Refusing to move unexpected remote output"
for name in sorted(expected):
    content = b"".join(v.read_file(remote_source + "/" + name))
    assert content == (source / name).read_bytes()
    (archive / name).write_bytes(content)
audit = {"archived_at": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
         "condition": condition, "reason": "GPU identity mismatch before any training update",
         "unchanged_worker_seconds": record["worker_seconds"],
         "files_sha256": {n: hashlib.sha256((archive / n).read_bytes()).hexdigest() for n in expected}}
(archive / "archive.json").write_text(json.dumps(audit, indent=2) + "\n")
remote_archive = remote_root + "/rejected-initializations/" + attempt
with v.batch_upload() as upload:
    for path in archive.iterdir():
        upload.put_file(path, remote_archive + "/" + path.name)
for path in archive.iterdir():
    assert b"".join(v.read_file(remote_archive + "/" + path.name)) == path.read_bytes()
for name in sorted(expected):
    assert b"".join(v.read_file(remote_source + "/" + name)) == (archive / name).read_bytes()
    v.remove_file(remote_source + "/" + name)
    (source / name).unlink()
assert not list(v.iterdir(remote_source))
print(json.dumps(audit, indent=2))
lock.close()
