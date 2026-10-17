from dataclasses import asdict
import io
import json
from pathlib import Path

import pytest

from gasp import eval_corpus as corpus
from gasp.data import import_eval, read_cache, sha256, write_cache
from gasp.models import Sample
from gasp.prompts import POLLUTER


def response(text="<polluted>0</polluted>", **changes):
    return {"id": "resp_test", "model": "gpt-6.1-sol", "status": "completed",
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}]} | changes


def contexts(tmp_path, config):
    source = tmp_path/"contexts.jsonl"
    rows = [{"id": f"q-a{alpha}", "question_id": "q", "question": "test e",
             "answer": "secret-gold", "prefix": "", "clean": "1", "split": "test",
             "alpha": alpha, "initial_clean_solved": True} for alpha in (0., .25, .5, .75)]
    write_cache(source, rows, {"kind": "eval_contexts",
        "config": asdict(config.with_changes(tiny=False)),
        "prompt_sha256": sha256(Path(corpus.__file__).with_name("prompts.py")),
        "subset_selection": "Unit-test fixture, no real corpus"})
    return source, tmp_path/"corruptions.jsonl"


def test_prepare_selects_all_correct_and_samples_a_fresh_trace(tmp_path, config, monkeypatch):
    source, output = tmp_path/"raw.jsonl", tmp_path/"contexts.jsonl"
    write_cache(source, [{"id": key, "question": f"test {key}", "answer": "1",
                         "reference_trace": "Never use this", "split": "test"}
                        for key in ("e", "f")], {"kind": "raw_eval"})
    calls = []
    # Four correct filter samples, a rejected fresh sample, a correct fresh
    # sample, then a question that fails the four-sample filter.
    outputs = iter([["<think>1</think><answer>1</answer>"]*4,
                    ["<think>0</think><answer>0</answer>"],
                    ["<think>1 + 0 = 1 + 0 = 1 + 0 = 1</think><answer>1</answer>"],
                    ["1", "1", "0", "1"]])
    def generate(self, model, prompt, n, limit, purpose, **kwargs):
        calls.append((n, purpose, kwargs["temperature"]))
        return [Sample(prompt, [4], text) for text in next(outputs)]
    monkeypatch.setattr(corpus.Rollouts, "generate", generate)
    corpus.prepare_eval_contexts(config, source, output)
    rows, manifest = read_cache(output, "eval_contexts")
    assert [r["alpha"] for r in rows] == [0., .25, .5, .75]
    assert {r["question_id"] for r in rows} == {"e"}
    assert all(r["clean_attempts"] == 2 and r["initial_clean_solved"] for r in rows)
    assert all("1 + 0 =" in r["clean_trace"] for r in rows)
    assert rows[0]["prefix"] == "" and rows[-1]["prefix"]
    assert calls == [(4, "eval_filter", .7), (1, "eval_clean_trace", .7),
                     (1, "eval_clean_trace", .7), (4, "eval_filter", .7)]
    meta = manifest["provenance"]
    assert meta["candidate_count"] == 2 and meta["selected_questions"] == 1
    assert meta["config"]["seed"] == config.eval_seed
    audit = [json.loads(s) for s in Path(str(output)+".audit.jsonl").read_text().splitlines()]
    assert [r["kept"] for r in audit] == [True, False]
    with pytest.raises(FileExistsError):
        corpus.prepare_eval_contexts(config, source, output)


def test_prepare_exhausted_trace_is_audited_without_reference_fallback(tmp_path, config, monkeypatch):
    source, output = tmp_path/"raw.jsonl", tmp_path/"contexts.jsonl"
    write_cache(source, [{"id": "e", "question": "test e", "answer": "1",
                         "reference_trace": "1", "split": "test"}], {"kind": "raw_eval"})
    def generate(self, model, prompt, n, limit, purpose, **kwargs):
        return [Sample(prompt, [4], "1" if n == 4 else "0") for _ in range(n)]
    monkeypatch.setattr(corpus.Rollouts, "generate", generate)
    with pytest.raises(RuntimeError, match="No correct nonempty"):
        corpus.prepare_eval_contexts(config.with_changes(max_clean_attempts=2), source, output)
    assert not output.exists()
    audit = json.loads(Path(str(output)+".audit.jsonl").read_text())
    assert audit["clean_outputs"] == ["0", "0"] and "error" in audit


def test_context_preparation_rejects_train_split_and_unresolved_model(tmp_path, config):
    source = tmp_path/"raw.jsonl"
    write_cache(source, [{"id": "q", "split": "train"}], {"kind": "raw_eval"})
    with pytest.raises(ValueError, match="test-split"):
        corpus.prepare_eval_contexts(config, source, tmp_path/"out")
    with pytest.raises(ValueError, match="resolved_config"):
        corpus.prepare_eval_contexts(config.with_changes(tiny=False, revision="main"),
                                     source, tmp_path/"out")


def test_generation_resumes_then_imports_the_same_saved_windows(tmp_path, config, monkeypatch):
    source, output = contexts(tmp_path, config)
    calls = []
    def request(payload):
        calls.append(payload)
        return response()
    monkeypatch.setattr(corpus, "request_corruption", request)
    first = corpus.generate_eval_corpus(source, output, max_requests=1)
    assert first == {"complete": False, "saved_contexts": 1, "total_contexts": 4}
    assert not output.exists()
    final = corpus.generate_eval_corpus(source, output)
    assert final["complete"] and len(calls) == 4
    assert all(p["model"] == "gpt-6.1-sol" and p["reasoning"] == {"effort": "low"} for p in calls)
    assert all("temperature" not in p and "top_p" not in p for p in calls)
    assert calls[0]["input"][1]["content"] == POLLUTER.format(q="test e", prefix="", continuation="1")
    assert "secret-gold" not in json.dumps(calls)
    meta = json.loads(Path(final["provenance"]).read_text())
    assert meta["corpus_sha256"] == sha256(output)
    assert "GPT-6.1 Sol" in meta["paper_deviation"]
    imported = tmp_path/"eval.jsonl"
    import_eval(config, output, imported, final["provenance"])
    rows, manifest = read_cache(imported, "eval")
    assert len(rows) == 4 and all(r["polluted"] == "0" for r in rows)
    assert all(r["generator"] == "gpt-6.1-sol" for r in rows)
    assert manifest["provenance"]["held_out"] is True
    with pytest.raises(FileExistsError):
        corpus.generate_eval_corpus(source, output)
    output.write_text(output.read_text().replace('"polluted": "0"', '"polluted": "1"'))
    with pytest.raises(ValueError, match="generation provenance"):
        import_eval(config, output, tmp_path/"tampered.jsonl", final["provenance"])


@pytest.mark.parametrize("bad", [response("missing tags"), response("<polluted></polluted>"),
                                 response(status="incomplete"), response(model="gpt-5"),
                                 response("<polluted>1</polluted>")])
def test_invalid_response_stops_and_is_not_billed_again(tmp_path, config, monkeypatch, bad):
    source, output = contexts(tmp_path, config)
    calls = []
    monkeypatch.setattr(corpus, "request_corruption", lambda payload: calls.append(payload) or bad)
    for _ in range(2):
        with pytest.raises(ValueError):
            corpus.generate_eval_corpus(source, output)
    assert len(calls) == 1 and not output.exists()
    assert json.loads(Path(str(output)+".responses/000000.json").read_text())["response"] == bad


def test_resume_refuses_changed_generation_settings(tmp_path, config, monkeypatch):
    source, output = contexts(tmp_path, config)
    monkeypatch.setattr(corpus, "request_corruption", lambda payload: response())
    corpus.generate_eval_corpus(source, output, max_requests=1)
    with pytest.raises(ValueError, match="Resume source/prompt/settings differ"):
        corpus.generate_eval_corpus(source, output, effort="high")
    with pytest.raises(ValueError, match="requested gpt-6.1-sol"):
        corpus.generate_eval_corpus(source, output, model="gpt-5")
    with pytest.raises(ValueError, match="reasoning effort"):
        corpus.generate_eval_corpus(source, output, effort="minimal")


def test_smoke_contexts_never_trigger_api_calls(tmp_path, config, monkeypatch):
    source, output = contexts(tmp_path, config)
    path = Path(str(source)+".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["provenance"]["config"]["tiny"] = True
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(corpus, "request_corruption", lambda payload: pytest.fail("API called"))
    with pytest.raises(ValueError, match="Synthetic smoke"):
        corpus.generate_eval_corpus(source, output)


def test_http_request_uses_responses_api_and_requires_an_environment_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        corpus.request_corruption({})
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-only")
    def urlopen(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert request.get_method() == "POST" and timeout == 180
        assert request.get_header("Authorization") == "Bearer unit-test-only"
        assert json.loads(request.data) == {"model": "gpt-6.1-sol"}
        return io.BytesIO(json.dumps(response()).encode())
    monkeypatch.setattr(corpus, "urlopen", urlopen)
    assert corpus.request_corruption({"model": "gpt-6.1-sol"}) == response()
