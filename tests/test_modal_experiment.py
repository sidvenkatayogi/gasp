import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest


def module(name):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def test_fixed_budget_config_and_schedule_freeze(config, tmp_path):
    worker = module("experiment_worker")
    a = worker.experiment_config(config, "unified")
    b = worker.experiment_config(config, "split")
    assert a.paired_hash() == b.paired_hash()
    assert (a.role_steps, a.block_steps, a.accumulation, a.group_size) == (50, 5, 1, 16)
    assert a.preparation_batch_questions == a.rollout_batch_size == 16
    assert a.prompt_tokens == 2048
    assert a.completion_tokens == a.corruption_tokens == a.repair_tokens == 1024
    worker.paired_plan(tmp_path, a, freeze=True)
    assert worker.paired_plan(tmp_path, b.with_changes(condition="unified"))["endpoint"] == 100
    with pytest.raises(ValueError, match="already frozen"):
        worker.paired_plan(tmp_path, a.with_changes(role_steps=45), freeze=True)


def test_incomplete_preparation_retains_progress_and_does_not_start_eval(config, tmp_path, monkeypatch):
    worker = module("experiment_worker")
    calls = []
    def prepare(cfg, source, output, **kwargs):
        calls.append((source, kwargs))
        Path(str(output) + ".progress.pt").write_bytes(b"durable progress")
        return {"complete": False, "selected_questions": 12, "reason": "time_limit"}
    monkeypatch.setattr(worker, "prepare_train", prepare)
    monkeypatch.setattr(worker, "prepare_eval_contexts", lambda *a, **k: pytest.fail("Eval must wait"))
    committed = []
    result = worker.prepare(tmp_path, config, worker.time.monotonic() + 1000,
                            lambda: committed.append(True), False)
    assert not result["complete"] and committed == [True]
    assert calls[0][1]["retained_target"] == 100 and calls[0][1]["max_candidates"] == 2000
    assert not calls[0][1]["resume"]
    worker.prepare(tmp_path, config, worker.time.monotonic() + 1000,
                   lambda: committed.append(True), True)
    assert calls[1][1]["resume"]


def test_cpu_preflight_requires_matching_semantic_review(tmp_path):
    runner = module("run_modal_experiment")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"q-a0"}\n')
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    audit = tmp_path / "audit.json"
    review = {"verdict": "pass", "scope": "all rows", "sample_ids": ["q-a0"],
              "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()}
    audit.write_text(json.dumps(review))
    archive = runner.build_inputs(tmp_path, corpus, provenance, audit)
    with zipfile.ZipFile(archive) as saved:
        assert set(saved.namelist()) == {"held-out-corruptions.jsonl",
            "held-out-corruptions.jsonl.provenance.json", "corpus-review.json"}
    corpus.write_text("changed")
    with pytest.raises(ValueError, match="semantic review"):
        runner.build_inputs(tmp_path, corpus, provenance, audit)


def test_lost_worker_charges_full_reservation_and_finished_worker_is_not_double_counted(tmp_path):
    runner = module("run_modal_experiment")
    launch = {"attempt_id": "prepare-1", "max_seconds": 3600}
    (tmp_path / "prepare-1.launch.json").write_text(json.dumps(launch))
    manifest = {"prior_gpu_seconds": 2880}
    assert runner.local_spend(tmp_path, manifest) == 2880 + 3600 + 60
    (tmp_path / "results").mkdir()
    (tmp_path / "results/gpu-budget.json").write_text(json.dumps({"attempts": [
        {"attempt": "prepare-1", "charged_seconds": 1000}]}))
    assert runner.local_spend(tmp_path, manifest) == 3880


def test_source_bundle_has_worker_but_never_uploads_env(tmp_path):
    runner = module("run_modal_experiment")
    # Reuse the repository's real explicit allowlist; it deliberately ignores .env.
    archive = tmp_path / "source.zip"
    hashes = runner.package_experiment(runner.ROOT, archive)
    with zipfile.ZipFile(archive) as saved:
        # Include every directly loaded test helper and the analysis CLI in the
        # remote source, not just files needed by the production worker itself.
        required = {"scripts/experiment_worker.py", "scripts/run_modal_experiment.py",
            "scripts/run_modal_benchmark.py", "scripts/run_colab_traces.py",
            "scripts/probe_cycle.py", "scripts/run_modal_cycle.py",
            "scripts/analyze_exploratory.py"}
        assert required <= set(saved.namelist())
        assert not any(part.startswith(".") for n in saved.namelist() for part in Path(n).parts)
        assert hashes["scripts/experiment_worker.py"] == hashlib.sha256(
            saved.read("scripts/experiment_worker.py")).hexdigest()
    runner.verify_launch_sources(runner.ROOT, archive)


@pytest.mark.parametrize("failure", [False, True])
def test_volume_commit_latency_is_logged_and_failures_propagate(tmp_path, failure):
    runner = module("run_modal_experiment")
    called = []
    def commit():
        called.append(True)
        if failure:
            raise RuntimeError("volume failed")
    path = tmp_path / "volume-commits.jsonl"
    measured = runner.timed_commit(commit, path)
    if failure:
        with pytest.raises(RuntimeError, match="volume failed"):
            measured()
    else:
        measured()
    event = json.loads(path.read_text())
    assert called == [True] and event["success"] is not failure
    assert event["phase"] == "volume_commit" and event["seconds"] >= 0


def test_result_extraction_cannot_escape(tmp_path):
    runner = module("run_modal_experiment")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as saved:
        saved.writestr("../bad.json", "{}")
    with zipfile.ZipFile(buffer) as saved, pytest.raises(ValueError, match="Unsafe archive"):
        runner.safe_extract(saved, tmp_path / "output")


def test_only_one_local_stage_can_launch(tmp_path):
    runner = module("run_modal_experiment")
    lock = runner.claim_local_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another stage"):
            runner.claim_local_lock(tmp_path)
    finally:
        lock.close()
    runner.claim_local_lock(tmp_path).close()


def test_real_tiny_paired_stages_share_baseline_and_persist_checkpoints(config, tmp_path):
    from gasp.data import make_smoke_data
    from gasp.trainer import Trainer
    worker = module("experiment_worker")
    cfg = config.with_changes(condition="unified", rollout_batch_size=2)
    make_smoke_data(tmp_path / "prepare", cfg)
    baseline = Trainer(cfg, tmp_path / "prepare/train.jsonl", tmp_path / "prepare/eval.jsonl",
                       tmp_path / "baseline")
    baseline.evaluate()
    worker.paired_plan(tmp_path, cfg, freeze=True)
    commits = []
    for condition in ("unified", "split"):
        if condition == "unified":
            stopped = worker.train(tmp_path, cfg, worker.time.monotonic(),
                lambda: commits.append(condition), False)
            assert not stopped["complete"] and stopped["step"] == 0
            assert not (tmp_path / "train/unified/eval").exists()
        result = worker.train(tmp_path, cfg.with_changes(condition=condition),
            worker.time.monotonic() + 1000, lambda: commits.append(condition), condition == "unified")
        assert result["complete"] and result["step"] == cfg.role_steps * 2
        assert result["role_counts"] == {"agent": cfg.role_steps, "polluter": cfg.role_steps}
        assert (tmp_path / "train" / condition / "checkpoint.pt").exists()
        assert (tmp_path / "train" / condition / "baseline-verification.json").exists()
    assert set(commits) == {"unified", "split"}
