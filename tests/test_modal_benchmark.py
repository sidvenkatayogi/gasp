import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest


def runner():
    spec = importlib.util.spec_from_file_location("modal_benchmark",
        Path(__file__).parents[1] / "scripts/run_modal_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_bundle_excludes_credentials_and_preserves_fixture(tmp_path):
    module = runner()
    root = tmp_path / "repo"
    # Minimal complete source tree, including tempting secret files in both the
    # root and a directory that is otherwise uploaded.
    required = ["scripts/probe_vllm.py", "scripts/run_colab_traces.py",
                "scripts/run_modal_benchmark.py", "pyproject.toml", "requirements-colab.txt",
                "data/full-v4/resolved_config.json", "data/raw/gsm_raw.jsonl",
                "data/raw/gsm_raw.jsonl.manifest.json", "data/raw/math_raw.jsonl",
                "data/raw/math_raw.jsonl.manifest.json", module.FIXTURE,
                "src/gasp/core.py", "src/gasp/templates/prompt.txt"]
    for name in required + [".env", "src/.env", "src/.private/key.json",
                           "src/project.egg-info/SOURCES.txt", "src/project.egg-info/PKG-INFO",
                           "src/project.dist-info/top_level.txt"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    archive = tmp_path / "source.zip"
    hashes = module.package_source(root, archive)
    with zipfile.ZipFile(archive) as saved:
        assert set(saved.namelist()) == set(required)
        assert hashes[module.FIXTURE] == hashlib.sha256(saved.read(module.FIXTURE)).hexdigest()


@pytest.mark.parametrize("success", [True, False])
def test_remote_failure_is_not_reported_as_success(tmp_path, success):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as saved:
        saved.writestr("worker.json", json.dumps({"success": success}))
    if success:
        assert runner().unpack_results(data.getvalue(), tmp_path)["success"]
    else:
        with pytest.raises(RuntimeError, match="benchmark failed"):
            runner().unpack_results(data.getvalue(), tmp_path)
        assert (tmp_path / "worker.json").exists()


def test_result_bundle_cannot_write_outside_destination(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as saved:
        saved.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unexpected result"):
        runner().unpack_results(data.getvalue(), tmp_path / "results")
    assert not (tmp_path / "escape").exists()
