import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def load_runner():
    spec = importlib.util.spec_from_file_location("trace_runner", Path(__file__).parents[1]/"scripts/run_colab_traces.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [None, "provision", "work"])
def test_gpu_is_released_on_success_and_failure(monkeypatch, failure):
    runner = load_runner()
    calls = []
    def colab(*args, **kwargs):
        calls.append(args[0])
        if failure == "provision" and args[0] == "new":
            raise RuntimeError("provision failed")
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(runner, "colab", colab)
    def execute():
        with runner.gpu_session("test"):
            if failure == "work":
                raise RuntimeError("work failed")
    if failure:
        with pytest.raises(RuntimeError):
            execute()
    else:
        execute()
    assert calls == ["new", "stop"]


def test_monitoring_timeout_does_not_abort_gpu_work(monkeypatch):
    runner = load_runner()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.colab("download", check=False, timeout=30)
    assert result.returncode == 124
    assert "timed out" in result.stderr
    with pytest.raises(subprocess.TimeoutExpired):
        runner.colab("upload", timeout=30)


def test_backup_downloads_checkpoint_before_audit(monkeypatch, tmp_path):
    runner = load_runner()
    calls = []
    monkeypatch.setattr(runner, "colab", lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.backup_contexts("test", tmp_path / "backup")
    assert calls[0][0][3].endswith("progress.pt")
    assert calls[1][0][3].endswith("audit.jsonl")
    assert all(call[1] == {"check": False, "timeout": 30} for call in calls)
