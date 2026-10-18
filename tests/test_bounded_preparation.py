import json
from pathlib import Path
import random

import pytest
import torch

from gasp import data, eval_corpus
from gasp.models import Rollouts, Sample, rng_state
from gasp.prompts import Prompts
from test_training import assert_tree_equal


def test_pinned_revision_resolves_without_network(config, monkeypatch):
    from huggingface_hub import HfApi
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(HfApi, "model_info", lambda *a, **k: pytest.fail("Pinned revision used network"))
    pinned = config.with_changes(tiny=False, revision="a" * 40)
    assert data.resolve_config(pinned) is pinned


def fixture(tmp_path, config, monkeypatch, kind, size=8, passing=None, retry_id=None):
    config = config.with_changes(preparation_batch_questions=4, rollout_batch_size=16)
    split, cache_kind = ("train", "raw_train") if kind == "train" else ("test", "raw_eval")
    rows = [{"id": f"q{i}", "question": str(i), "answer": "1", "split": split,
             "reference_trace": "Never use this reference"} for i in range(size)]
    source = tmp_path/f"{kind}-raw.jsonl"
    data.write_cache(source, rows, {"kind": cache_kind})
    order = list(range(size))
    random.Random(42 if kind == "train" else config.eval_seed).shuffle(order)
    monkeypatch.setattr(Prompts, "agent", lambda self, question: [int(question)])
    calls, attempts = [], {}

    def generate_many(self, model, prompts, n, limit, purpose, **kwargs):
        calls.append(([prompt[0] for prompt in prompts], n, purpose, kwargs["temperature"]))
        count = self.counts.setdefault(purpose, {"sequences": 0, "generated_tokens": 0,
                                               "prompt_tokens": 0, "allocated_tokens": 0})
        groups = []
        for prompt in prompts:
            question = prompt[0]
            attempts[question] = attempts.get(question, 0) + (purpose == "eval_clean_trace")
            samples = []
            for index in range(n):
                noise = int(torch.randint(0, 10000, (1,)))
                success = passing is None or question in passing
                if purpose == "eval_clean_trace":
                    success = not (question == retry_id and attempts[question] == 1)
                elif index != 2:
                    success = True  # Failed filters are exactly [1, 1, 0, 1].
                text = f"<think>{noise} + 0 = 1 + 0 = 1</think><answer>{int(success)}</answer>"
                samples.append(Sample(prompt, [4], text))
                count["sequences"] += 1
                count["generated_tokens"] += 1
                count["prompt_tokens"] += len(prompt)
                count["allocated_tokens"] += limit
            groups.append(samples)
        return groups

    monkeypatch.setattr(Rollouts, "generate_many", generate_many)
    prepare = data.prepare_train if kind == "train" else eval_corpus.prepare_eval_contexts
    return config, source, order, calls, prepare


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_target_keeps_first_eligible_questions_in_recorded_order(tmp_path, config, monkeypatch, kind):
    config, source, order, calls, prepare = fixture(tmp_path, config, monkeypatch, kind)
    # All four filters pass; only the first two may enter the target cache.
    output = tmp_path/"prepared.jsonl"
    result = prepare(config, source, output, retained_target=2, max_candidates=6)
    assert result["complete"] and result["selected_questions"] == 2
    assert result["processed"] == 4 and result["reason"] == "target_reached"
    rows, manifest = data.read_cache(output)
    kept = [r["id"] for r in rows] if kind == "train" else list(dict.fromkeys(r["question_id"] for r in rows))
    assert kept == [f"q{i}" for i in order[:2]]
    assert not any("reference_trace" in row for row in rows)
    recorded = json.loads(Path(str(output)+".candidate_order.json").read_text())
    assert recorded["candidate_ids"] == [f"q{i}" for i in order[:6]]
    assert recorded["seed"] == (42 if kind == "train" else config.eval_seed)
    assert manifest["provenance"]["candidate_order_sha256"] == data.sha256(str(output)+".candidate_order.json")
    assert calls[0] == (order[:4], 4, f"{kind}_filter", .7)
    assert result["counts"][f"{kind}_filter"]["sequences"] == 16
    audit = [json.loads(line) for line in Path(str(output)+".audit.jsonl").read_text().splitlines()]
    assert [row["kept"] for row in audit] == [True, True, False, False]
    assert all(row["reason"] == "target_already_reached_in_batch" for row in audit[2:])
    if kind == "eval":
        assert calls[1] == (order[:2], 1, "eval_clean_trace", .7)
        assert [row["alpha"] for row in rows] == [0., .25, .5, .75]*2
        assert all(row["initial_clean_solved"] for row in rows)
    with pytest.raises(FileExistsError):
        prepare(config, source, output, retained_target=2, max_candidates=6)


def test_eval_retries_only_failed_fresh_traces_without_changing_question_order(tmp_path, config, monkeypatch):
    order = list(range(8))
    random.Random(config.eval_seed).shuffle(order)
    config, source, order, calls, prepare = fixture(tmp_path, config, monkeypatch, "eval", retry_id=order[0])
    output = tmp_path/"contexts.jsonl"
    prepare(config, source, output, retained_target=2, max_candidates=4)
    assert calls[1:] == [(order[:2], 1, "eval_clean_trace", .7),
                         ([order[0]], 1, "eval_clean_trace", .7)]
    rows, _ = data.read_cache(output)
    assert [row["question_id"] for row in rows] == [f"q{order[0]}"]*4+[f"q{order[1]}"]*4
    assert [row["clean_attempts"] for row in rows] == [2]*4+[1]*4
    audit = [json.loads(line) for line in Path(str(output)+".audit.jsonl").read_text().splitlines()]
    assert len(audit[0]["clean_outputs"]) == 2
    assert rows[0]["clean_trace"] not in audit[0]["filter_outputs"]


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_unmet_target_never_writes_a_partial_final_cache(tmp_path, config, monkeypatch, kind):
    config, source, order, calls, prepare = fixture(tmp_path, config, monkeypatch, kind, passing=set())
    output = tmp_path/"short.jsonl"
    result = prepare(config, source, output, retained_target=2, max_candidates=3)
    assert result["complete"] is False and result["reason"] == "candidate_limit"
    assert result["processed"] == 3 and result["selected_questions"] == 0
    assert result["counts"][f"{kind}_filter"]["sequences"] == 12
    assert not output.exists() and not Path(str(output)+".manifest.json").exists()
    assert Path(str(output)+".progress.pt").exists()
    assert json.loads(Path(str(output)+".status.json").read_text()) == result
    audit = [json.loads(line) for line in Path(str(output)+".audit.jsonl").read_text().splitlines()]
    assert len(audit) == 3 and not any(row["kept"] for row in audit)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_time_bounded_resume_preserves_rng_outputs_and_committed_audit(tmp_path, config, monkeypatch, kind):
    config, source, _, _, prepare = fixture(tmp_path, config, monkeypatch, kind)
    config = config.with_changes(preparation_batch_questions=2)
    expected, actual = tmp_path/"expected.jsonl", tmp_path/"actual.jsonl"
    prepare(config, source, expected, retained_target=4, max_candidates=6)
    expected_rng = rng_state()
    first = prepare(config, source, actual, retained_target=4, max_candidates=6, max_seconds=1e-12)
    assert first["complete"] is False and first["reason"] == "time_limit"
    assert first["processed"] == first["selected_questions"] == 2 and not actual.exists()
    with pytest.raises(ValueError, match="config/source/code"):
        prepare(config, source, actual, retained_target=5, max_candidates=6, resume=True)
    final = prepare(config, source, actual, retained_target=4, max_candidates=6, resume=True)
    assert final["complete"] and final["counts"] == data.read_cache(expected)[1]["provenance"]["counts"]
    assert actual.read_bytes() == expected.read_bytes()
    assert Path(str(actual)+".audit.jsonl").read_bytes() == Path(str(expected)+".audit.jsonl").read_bytes()
    assert_tree_equal(expected_rng, rng_state())


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_offline_batch_size_is_independent_of_live_rollout_chunk_size(tmp_path, config, monkeypatch, kind):
    config, source, order, calls, prepare = fixture(tmp_path, config, monkeypatch, kind)
    config = config.with_changes(preparation_batch_questions=8, rollout_batch_size=4)
    prepare(config, source, tmp_path/"out", retained_target=1, max_candidates=8)
    assert calls[0][0] == order and calls[0][1] == 4


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_interrupted_batch_is_replayed_from_committed_rng(tmp_path, config, monkeypatch, kind):
    config, source, _, _, prepare = fixture(tmp_path, config, monkeypatch, kind)
    config = config.with_changes(preparation_batch_questions=2)
    expected, actual = tmp_path/"expected.jsonl", tmp_path/"actual.jsonl"
    prepare(config, source, expected, retained_target=4, max_candidates=6)
    expected_rng = rng_state()
    original = Rollouts.generate_many
    control = {"filter_calls": 0, "interrupt": True}

    def interrupt_after_generation(self, model, prompts, n, limit, purpose, **kwargs):
        result = original(self, model, prompts, n, limit, purpose, **kwargs)
        if purpose == f"{kind}_filter":
            control["filter_calls"] += 1
            if control["interrupt"] and control["filter_calls"] == 2:
                raise KeyboardInterrupt("Interrupted after advancing RNG")
        return result

    monkeypatch.setattr(Rollouts, "generate_many", interrupt_after_generation)
    with pytest.raises(KeyboardInterrupt):
        prepare(config, source, actual, retained_target=4, max_candidates=6)
    saved = torch.load(str(actual)+".progress.pt", weights_only=False)
    assert saved["cursor"] == 2
    assert not actual.exists()
    control["interrupt"] = False
    prepare(config, source, actual, retained_target=4, max_candidates=6, resume=True)
    assert actual.read_bytes() == expected.read_bytes()
    assert Path(str(actual)+".audit.jsonl").read_bytes() == Path(str(expected)+".audit.jsonl").read_bytes()
    archived = list(tmp_path.glob("actual.jsonl.audit.jsonl.interrupted-*"))
    assert len(archived) == 1 and "KeyboardInterrupt" in archived[0].read_text()
    assert_tree_equal(expected_rng, rng_state())


def test_bounded_eval_trace_exhaustion_is_a_failure_not_a_skipped_question(tmp_path, config, monkeypatch):
    config, source, _, _, prepare = fixture(tmp_path, config, monkeypatch, "eval")
    config = config.with_changes(max_clean_attempts=2)
    original = Rollouts.generate_many

    def failed_fresh_trace(self, model, prompts, n, limit, purpose, **kwargs):
        groups = original(self, model, prompts, n, limit, purpose, **kwargs)
        if purpose == "eval_clean_trace":
            for group in groups:
                group[0].text = "<think>1 + 1 = 0</think><answer>0</answer>"
        return groups

    monkeypatch.setattr(Rollouts, "generate_many", failed_fresh_trace)
    output = tmp_path/"failed.jsonl"
    with pytest.raises(RuntimeError, match="No correct nonempty"):
        prepare(config, source, output, retained_target=2, max_candidates=4)
    assert not output.exists()
    audit = [json.loads(line) for line in Path(str(output)+".audit.jsonl").read_text().splitlines()]
    assert len(audit[-1]["clean_outputs"]) == 2 and "error" in audit[-1]
    assert json.loads(Path(str(output)+".status.json").read_text())["reason"] == "error"
