import json
from pathlib import Path

import pytest
from gasp.config import Config
from gasp.data import import_eval, read_cache, validate_eval_provenance
from gasp.models import Sample, token_logps
from gasp.prompts import Prompts
from gasp.verification import correct, pollution, repair_target


class TextTokenizer:
    """Lossless tokenizer for testing the actual non-smoke prompt construction."""
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages) + "\nAssistant:<think>\n"


@pytest.mark.parametrize("prefix,window,expected", [
    ("The value is 10", "24.", "The value is 1024."),
    ("We calcul", "ate 1 + 1 = 3", "We calculate 1 + 1 = 3"),
    ("We calculate", " 1 + 1 = 3", "We calculate 1 + 1 = 3"),
    ("We calculate\n", "1 + 1 = 3", "We calculate\n1 + 1 = 3"),
    ("", "1 + 1 = 3", "<think>\n1 + 1 = 3"),
])
def test_window_boundary_is_preserved(prefix, window, expected):
    tok=TextTokenizer()
    p=Prompts(tok, Config(prompt_tokens=4096))
    extracted, tagged=pollution(f"<polluted>{window}</polluted>")
    assert extracted==window and tagged
    assert tok.decode(p.agent("Add.", prefix, extracted)).endswith(expected)


@pytest.mark.parametrize("role,args", [
    ("agent", ("Add.", "The unmodified shared prefix. ", "1+1=3")),
    ("polluter", ("Add.", "The unmodified shared prefix. ", "1+1=2")),
    ("repair", ("Add.", "The unmodified shared prefix. ", "1+1=2", "1+1=3")),
])
def test_prompt_overflow_never_silently_changes_shared_prefix(role, args):
    tok=TextTokenizer()
    p=Prompts(tok, Config(prompt_tokens=10000))
    ids=getattr(p, role)(*args)
    assert args[1] in tok.decode(ids)
    p=Prompts(tok, Config(prompt_tokens=len(ids)-1))
    with pytest.raises(ValueError, match="refusing to truncate the shared reasoning prefix"):
        getattr(p, role)(*args)


def test_no_semantic_or_length_filters_on_windows_and_repairs():
    tok=TextTokenizer()
    for text in ("The difference is a-b.", "Same text.", "x"*1100):
        assert pollution(f"<polluted>{text}</polluted>")== (text, True)
        assert pollution(text)==(text, False)
    # Appendix B allows a final answer when the context forces one. The prompt
    # also requests 1-4 sentences; it does not specify dropping longer samples.
    repair=r"The corrected answer is \boxed{2}. " + "Continue. "*5 + "x"*300
    assert tok.decode(repair_target(repair, tok))==repair
    assert repair_target("thinking</think>", tok)==[]


@pytest.mark.parametrize("raw,prompt,expected", [
    ("Compare the CLEAN window before writing.", "Assistant:<think>\n", ""),
    ("<think>Still comparing.", "Assistant:", ""),
    ("Compare. </think>Actually, 1 + 1 = 2.", "Assistant:<think>\n",
     "Actually, 1 + 1 = 2."),
    ("Actually, 1 + 1 = 2.", "Assistant:", "Actually, 1 + 1 = 2."),
])
def test_visible_targets_respect_prompt_opened_thinking(raw, prompt, expected):
    tok=TextTokenizer()
    assert tok.decode(repair_target(raw, tok, prompt))==expected
    assert pollution(raw, prompt)==(expected, False)


def test_trainer_does_not_clone_unfinished_privileged_reasoning(factory, monkeypatch):
    t=factory("split")
    t.roles.refresh("agent")
    original=t.rollouts.generate
    original_decode=t.tokenizer.decode
    # The target tokenizer puts the opening <think> in the prompt, so the
    # generated continuation itself has no opening marker to inspect.
    repair_prompt=[31, 31, 31]
    def decode(ids, **kwargs):
        return "Assistant:<think>\n" if ids==repair_prompt else original_decode(ids, **kwargs)
    def generate(model, prompt, n, limit, purpose, **kwargs):
        if purpose=="repair_generation":
            return [Sample(repair_prompt, [3], "Compare the CLEAN and POLLUTED windows.")]
        return original(model, prompt, n, limit, purpose, **kwargs)
    monkeypatch.setattr(t.tokenizer, "decode", decode)
    monkeypatch.setattr(t.rollouts, "generate", generate)
    result=t.agent_context({"question":"train a", "answer":"1"}, "", "1")
    assert result["repair_target_ids"]==[] and result["guidance_nll"]==0
    assert t.loss_tokens["repair"]==0
    assert t.loss_tokens["agent"]==t.cfg.group_size


def test_clean_context_retries_current_policy_and_never_uses_cache(factory, monkeypatch):
    t=factory("split")
    row={"id":"retry", "question":"train a", "answer":"1", "initial_trace":"8"}
    outputs=iter(["0", "1"])
    calls=[]
    def generate(model, prompt, n, limit, purpose, **kwargs):
        calls.append((model, n))
        text=next(outputs)
        return [Sample(prompt, t.tokenizer.encode(text), text)]
    monkeypatch.setattr(t.rollouts, "generate", generate)
    prefix, window, attempts=t.make_context(row, 0.)
    assert prefix=="" and window=="1" and attempts==2
    assert calls==[(t.roles.agent,1), (t.roles.agent,1)]


def test_clean_retry_exhaustion_does_not_advance_optimizer(factory, monkeypatch):
    t=factory("split")
    t.cfg=t.cfg.with_changes(max_clean_attempts=2)
    row={"id":"unsolved", "question":"train a", "answer":"1", "initial_trace":"1"}
    monkeypatch.setattr(t, "next_row", lambda:(row, 0.))
    monkeypatch.setattr(t.rollouts, "generate", lambda model,prompt,*args,**kw:
                        [Sample(prompt,[3],"0")])
    with pytest.raises(RuntimeError, match="No correct on-policy clean trace"):
        t.step()
    assert t.global_step==0 and t.role_counts=={"agent":0,"polluter":0}
    assert not t.optimizer.state and t.scheduler.last_epoch==0


@pytest.mark.parametrize("role", ["agent", "polluter"])
def test_unformatted_corruption_does_not_mask_training(factory, monkeypatch, role):
    t=factory("split")
    t.role=role
    t.roles.refresh(role)
    scored=[0]
    def generate(model, prompt, n, limit, purpose, old=False, **kwargs):
        results=[]
        for i in range(n):
            token=3+i%2
            text=str(i%2)
            if purpose in ("agent_corruption", "polluter_candidate"):
                text="The difference is a-b."  # Previously rejected, no tags/digits.
            elif purpose=="polluter_agent_scoring":
                text=str(scored[0]%2)
                scored[0]+=1
            elif purpose=="repair_generation":
                text="1"; token=4
            old_logps=token_logps(model,prompt,[token],t.cfg).detach() if old else None
            results.append(Sample(prompt,[token],text,old_logps))
        return results
    monkeypatch.setattr(t.rollouts, "generate", generate)
    entry=getattr(t, role+"_context")({"question":"train a","answer":"1"}, "", "1")
    assert not any(entry["polluted_tag_present"])
    assert entry["informative_samples"]==t.cfg.group_size
    assert t.loss_tokens[role]==t.cfg.group_size
    assert all(p.grad is None for p in t.roles.opponent.parameters())
    if role=="agent":
        assert t.loss_tokens["repair"]==1 and entry["guidance_nll"]>0


def provenance():
    return {"protocol":"unit-test-fixture", "generator_model":"gpt-5",
            "generator_revision":"fixture", "source_url":"https://example.com/test-fixture",
            "generation_settings":{"temperature":0.7},
            "subset_selection":"initial checkpoint K=4, T=0.7, all correct",
            "held_out":True}


def test_eval_import_preserves_external_windows_and_refuses_substitutes(config, tmp_path):
    source=tmp_path/"external.jsonl"; meta=tmp_path/"provenance.json"
    row={"id":"q-a0", "question_id":"q", "question":"test e", "answer":"1",
         "prefix":"", "polluted":"0", "initial_clean_solved":True}
    source.write_text(json.dumps(row)+"\n")
    meta.write_text(json.dumps(provenance()))
    import_eval(config,source,tmp_path/"eval.jsonl",meta)
    rows, manifest=read_cache(tmp_path/"eval.jsonl","eval")
    assert rows[0]["polluted"]=="0" and rows[0]["generator"]=="gpt-5"
    assert manifest["provenance"]["held_out"] is True
    # Missing corruption is an error, never a numeral-edit fallback.
    row["polluted"]=""
    source.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="no synthetic fallback"):
        import_eval(config,source,tmp_path/"bad.jsonl",meta)
    for changes in ({"generator_model":"deepseek-ai/base"}, {"held_out":False}):
        with pytest.raises(ValueError,match="GPT-5|held_out"):
            validate_eval_provenance(provenance() | changes)


def test_real_symbolic_verifier():
    assert correct(r"\boxed{\frac{1}{2}}", "0.5", "math_verify")==1
    assert correct(r"<answer>x+x</answer>", "2x", "math_verify")==1
    assert correct(r"\boxed{3}", "2", "math_verify")==0
    assert correct("Unfinished calculation: 2", "2", "math_verify")==0


def test_pilot_only_shortens_run_not_algorithm():
    root=Path(__file__).parents[1]/"configs"
    full=json.loads((root/"full.json").read_text())
    pilot=json.loads((root/"pilot.json").read_text())
    differing={key for key in full if full[key]!=pilot[key]}
    assert differing=={"label","role_steps","eval_every","checkpoint_every"}
    assert full["condition"]=="split" and full["tuning"]=="full"
    assert full["group_size"]==64 and full["completion_tokens"]==1024


def test_real_training_refuses_synthetic_evaluation_before_loading_model(config, tmp_path):
    from gasp.data import make_smoke_data
    from gasp.trainer import Trainer
    make_smoke_data(tmp_path/"data", config)
    with pytest.raises(ValueError, match="Missing evaluation provenance"):
        Trainer(config.with_changes(tiny=False), tmp_path/"data/train.jsonl",
                tmp_path/"data/eval.jsonl", tmp_path/"run")


def test_old_checkpoint_protocol_cannot_be_silently_resumed(factory):
    import torch
    from gasp.data import sha256, write_json
    t=factory()
    path=t.save()
    state=torch.load(path, weights_only=False)
    state["schema"]=1
    torch.save(state, path)
    write_json(str(path)+".sha256.json", {"sha256":sha256(path)})
    with pytest.raises(ValueError, match="Old training protocol"):
        t.resume(path)
