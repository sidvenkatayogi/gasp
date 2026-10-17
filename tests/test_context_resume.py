import json
from pathlib import Path

import pytest
import torch

from gasp import eval_corpus as corpus
from gasp.data import read_cache, write_cache
from gasp.models import Sample, rng_state
from test_training import assert_tree_equal


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("batch_questions", [1, 2])
def test_preparation_resume_preserves_outputs_rng_and_audit(tmp_path, config, monkeypatch, interrupted, batch_questions):
    config = config.with_changes(preparation_batch_questions=batch_questions)
    source = tmp_path / "raw.jsonl"
    write_cache(source, [{"id": f"q{i}", "question": f"test {i}", "answer": "1", "split": "test"}
                        for i in range(3)], {"kind": "raw_eval"})
    control = {"calls": 0, "interrupt": False}

    def generate(self, model, prompt, n, limit, purpose, **kwargs):
        control["calls"] += 1
        noise = torch.randint(0, 10000, (n,)).tolist()
        if control["interrupt"] and control["calls"] == 3:
            raise KeyboardInterrupt("simulated runtime interruption")
        counts = self.counts.setdefault(purpose, {"sequences": 0})
        counts["sequences"] += n
        return [Sample(prompt, [4], f"<think>{value} + 1 = 1 + 0 = 1</think><answer>1</answer>")
                for value in noise]

    monkeypatch.setattr(corpus.Rollouts, "generate", generate)
    expected = tmp_path / "expected.jsonl"
    corpus.prepare_eval_contexts(config, source, expected)
    expected_rng = rng_state()
    actual = tmp_path / "actual.jsonl"
    control.update(calls=0, interrupt=interrupted)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            corpus.prepare_eval_contexts(config, source, actual)
    else:
        result = corpus.prepare_eval_contexts(config, source, actual, max_seconds=1e-12)
        assert result == {"complete": False, "processed": batch_questions, "total": 3}
    assert not actual.exists()
    control["interrupt"] = False
    corpus.prepare_eval_contexts(config, source, actual, resume=True)
    assert read_cache(actual) == read_cache(expected)
    assert Path(str(actual)+".audit.jsonl").read_bytes() == Path(str(expected)+".audit.jsonl").read_bytes()
    assert_tree_equal(expected_rng, rng_state())
    if interrupted:
        archived = list(tmp_path.glob("actual.jsonl.audit.jsonl.interrupted-*"))
        assert len(archived) == 1 and "KeyboardInterrupt" in archived[0].read_text()


def test_preparation_resume_rejects_mismatch_and_missing_audit(tmp_path, config, monkeypatch):
    source, output = tmp_path / "raw.jsonl", tmp_path / "out.jsonl"
    write_cache(source, [{"id": "q", "question": "test e", "answer": "1", "split": "test"}],
                {"kind": "raw_eval"})
    with pytest.raises(FileNotFoundError, match="checkpoint and audit"):
        corpus.prepare_eval_contexts(config, source, output, resume=True)
    monkeypatch.setattr(corpus.Rollouts, "generate", lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("failure before first committed question")))
    with pytest.raises(RuntimeError):
        corpus.prepare_eval_contexts(config, source, output)
    with pytest.raises(ValueError, match="config/source/code"):
        corpus.prepare_eval_contexts(config.with_changes(eval_seed=123), source, output, resume=True)
