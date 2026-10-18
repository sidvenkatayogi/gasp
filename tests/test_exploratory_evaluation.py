import hashlib
import json
import random

import numpy as np
import pytest
import torch

from gasp.evaluation import analyze_paired, evaluate, verify_shared_baseline
from gasp.models import Rollouts, rng_state
from test_training import assert_tree_equal


def four_alpha_rows(trainer):
    return [dict(row, id=row["id"] + "-" + str(alpha), alpha=alpha,
                 initial_clean_solved=True,
                 corrupt_prompt_ids=row["corrupt_prompt_ids"] + [3] * index)
            for index, alpha in enumerate((0., .25, .5, .75)) for row in trainer.eval_rows]


def evaluate_rows(trainer, rows, output, config=None):
    return evaluate(trainer.roles.agent, trainer.tokenizer, config or trainer.cfg,
                    rows, trainer.eval_manifest, output, 0, trainer.run_manifest)


def test_batched_evaluation_maps_interleaved_questions_and_preserves_rng(factory, tmp_path, monkeypatch):
    trainer = factory()
    rows = four_alpha_rows(trainer)
    serial = trainer.cfg.with_changes(tiny=False, completion_tokens=4)
    batched = serial.with_changes(rollout_batch_size=3)
    calls = []
    original = Rollouts.generate_many

    def record(self, model, prompts, n, limit, purpose, **kwargs):
        calls.append((purpose, len(prompts), kwargs["temperature"]))
        return original(self, model, prompts, n, limit, purpose, **kwargs)

    monkeypatch.setattr(Rollouts, "generate_many", record)
    before = rng_state()
    expected = evaluate_rows(trainer, rows, tmp_path/"serial", serial)
    actual = evaluate_rows(trainer, rows, tmp_path/"batched", batched)
    assert_tree_equal(before, rng_state())
    assert actual["metrics"] == expected["metrics"]
    assert actual["counts"] == expected["counts"]
    assert actual["n_clean"] == 4 and actual["n_corrupted"] == 16
    assert calls == [("eval_clean", n, 0.) for n in (3, 1)] + [
        ("eval_corrupted", n, 0.) for n in (3, 3, 3, 3, 3, 1)]
    assert (tmp_path/"serial/eval/step-000000/predictions.jsonl").read_bytes() == (
        tmp_path/"batched/eval/step-000000/predictions.jsonl").read_bytes()
    assert trainer.roles.agent.training
    assert all(p.grad is None for p in trainer.roles.agent.parameters())


def test_batched_evaluation_restores_rng_on_failure(factory, tmp_path, monkeypatch):
    trainer = factory()
    before = rng_state()

    def fail(*args, **kwargs):
        random.random(); np.random.random(); torch.rand(1)
        raise RuntimeError("failed batch")

    monkeypatch.setattr(Rollouts, "generate_many", fail)
    with pytest.raises(RuntimeError, match="failed batch"):
        evaluate_rows(trainer, trainer.eval_rows, tmp_path, trainer.cfg.with_changes(rollout_batch_size=3))
    assert_tree_equal(before, rng_state())


def test_shared_baseline_checks_identity_predictions_and_exact_batch_shapes(factory, tmp_path, monkeypatch):
    trainer = factory()
    rows = four_alpha_rows(trainer)
    cfg = trainer.cfg.with_changes(tiny=False, rollout_batch_size=3, completion_tokens=4)
    evaluate_rows(trainer, rows, tmp_path/"baseline", cfg)
    baseline = tmp_path/"baseline/eval/step-000000"
    before = rng_state()
    verification = verify_shared_baseline(trainer.roles.agent, trainer.tokenizer, cfg,
        rows, trainer.eval_manifest, trainer.run_manifest, baseline, tmp_path/"run")
    assert_tree_equal(before, rng_state())
    assert verification["passed"]
    assert verification["counts"]["eval_clean"]["sequences"] == 3
    assert verification["counts"]["eval_corrupted"]["sequences"] == 3
    assert not (tmp_path/"run/eval").exists()  # No invented second full baseline.
    with pytest.raises(ValueError, match="paired_config_sha256"):
        verify_shared_baseline(trainer.roles.agent, trainer.tokenizer, cfg.with_changes(eval_seed=3),
            rows, trainer.eval_manifest, trainer.run_manifest, baseline, tmp_path/"bad")
    original = Rollouts.generate_many

    def change_prediction(*args, **kwargs):
        groups = original(*args, **kwargs)
        groups[0][0].text += " changed"
        return groups

    monkeypatch.setattr(Rollouts, "generate_many", change_prediction)
    with pytest.raises(ValueError, match="spot-check prediction mismatch"):
        verify_shared_baseline(trainer.roles.agent, trainer.tokenizer, cfg,
            rows, trainer.eval_manifest, trainer.run_manifest, baseline, tmp_path/"bad")
    assert_tree_equal(before, rng_state())


def write_prediction_artifact(path, condition, question_scores, clean_scores, step):
    path.mkdir()
    predictions = [{"id": f"q{qid}-{alpha}", "question_id": f"q{qid}", "question": f"Question {qid}",
        "answer": "1", "prediction": "1" if score else "0", "correct": score,
        "clean": {"prediction": str(clean_scores[qid]), "correct": clean_scores[qid]},
        "alpha": alpha, "initial_clean_solved": True, "eval_sha256": "corpus"}
        for qid, score in enumerate(question_scores) for alpha in (0., .25, .5, .75)]
    metrics = {"paired_config_sha256": "paired", "eval_sha256": "corpus", "train_sha256": "train",
        "protocol": "fixed", "source_sha256": "source", "tokenizer_sha256": "tokenizer",
        "packages": {}, "gpu": None, "prompt_ids_sha256": "prompts", "model_sha256": condition,
        "step": step, "condition": condition, "config": {"role_steps": 50}, "seed": 42}
    save_prediction_artifact(path, metrics, predictions)
    return path


def save_prediction_artifact(path, metrics, predictions):
    text = "".join(json.dumps(p) + "\n" for p in predictions)
    (path/"predictions.jsonl").write_text(text)
    metrics["predictions_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    (path/"metrics.json").write_text(json.dumps(metrics))


def analysis_inputs(tmp_path):
    return [write_prediction_artifact(tmp_path/name, name, scores, clean, step)
            for name, scores, clean, step in (("initial", [0, 0], [1, 1], 0),
                ("unified", [0, 1], [0, 1], 100), ("split", [1, 0], [1, 0], 100))]


def test_paired_bootstrap_resamples_questions_with_four_correlated_alphas(tmp_path):
    paths = analysis_inputs(tmp_path)
    before = rng_state()
    result = analyze_paired(*paths, tmp_path/"report")
    assert_tree_equal(before, rng_state())
    assert result["n_questions"] == 2 and result["n_corruptions"] == 8
    delta = result["paired_differences"]["split_minus_unified"]
    # Each question's four attacks are perfectly correlated. Resampling eight
    # rows independently would incorrectly narrow this question-cluster interval.
    assert delta["corrupted_accuracy"] == {"estimate": 0., "ci95": [-1., 1.]}
    assert delta["clean_selected_subset_accuracy"] == delta["corrupted_accuracy"]
    assert result["paired_differences"]["split_minus_initial"]["corrupted_accuracy"] == {
        "estimate": .5, "ci95": [0., 1.]}
    assert result == analyze_paired(*paths, tmp_path/"repeated")
    assert result["bootstrap"] == {"unit": "question", "samples": 10000, "seed": 10002,
                                    "interval": "paired percentile 95%"}


@pytest.mark.parametrize("damage,match", [("hash", "hash mismatch"),
    ("corpus", "eval_sha256"), ("order", "identities/order"), ("alpha", "four declared alpha"),
    ("clean", "Inconsistent repeated clean"), ("endpoint", "fixed complete")])
def test_paired_analysis_rejects_mismatches(tmp_path, damage, match):
    paths = analysis_inputs(tmp_path)
    if damage == "hash":
        with (paths[1]/"predictions.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        for path in paths if damage == "alpha" else paths[1:2]:
            metrics = json.loads((path/"metrics.json").read_text())
            predictions = [json.loads(line) for line in (path/"predictions.jsonl").read_text().splitlines()]
            if damage == "corpus":
                metrics["eval_sha256"] = "different"
                for prediction in predictions:
                    prediction["eval_sha256"] = "different"
            elif damage == "order":
                predictions.reverse()
            elif damage == "alpha":
                predictions[0]["alpha"] = .25
            elif damage == "clean":
                predictions[0]["clean"]["correct"] = 1
            elif damage == "endpoint":
                metrics["step"] = 90
            save_prediction_artifact(path, metrics, predictions)
    with pytest.raises(ValueError, match=match):
        analyze_paired(*paths, tmp_path/"report")
