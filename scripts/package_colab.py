"""Create a source-only Colab archive; no credentials, .venv, weights or checkpoints."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1]
files=[]
for directory in ("src","configs","docs","notebooks","scripts","tests"):
    files.extend(p for p in (root/directory).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix != ".pyc")
files += [root/name for name in ("README.md","pyproject.toml","requirements-colab.txt")]
files += [root/"artifacts"/name for name in ("VALIDATION.md","local-environment.json","audit-tokenizer-checks.json") if (root/"artifacts"/name).exists()]
destination=root/"dist/gasp-colab.zip"
destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        z.write(path,"gasp/"+str(path.relative_to(root)))
    # Always create a writable artifacts directory for notebook temporary outputs.
    z.writestr("gasp/artifacts/.keep", "")
manifest={"archive_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
          "files":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(root/"dist/manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(destination, destination.stat().st_size, "bytes")
