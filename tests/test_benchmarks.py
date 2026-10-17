import hashlib
import json
import pytest

from gasp.benchmarks import parse_diagnosis, score_diagnosis, prepare_rup, prepare_mr
from gasp.data import read_cache
from gasp.models import Rollouts, load_model
from gasp.prompts import Prompts
from gasp.verification import repair_target, pollution


def test_diagnosis_needs_explanation_judgement(tmp_path):
    prediction="Final Judgement: wrong\nFirst Error Step: 2\nError Analysis: addition is incorrect"
    row={"id":"a","gold_correct":False,"gold_step":"2","prediction":prediction}|parse_diagnosis(prediction)
    assert score_diagnosis([row])["step_accuracy"]==1
    assert score_diagnosis([row])["ACCreason"] is None
    judge={"judge_provenance":"manual-test","decisions":[{"id":"a",
        "prediction_sha256":hashlib.sha256(prediction.encode()).hexdigest(),"explanation_correct":False}]}
    assert score_diagnosis([row],judge)["ACCreason"]==0
    judge["decisions"][0]["explanation_correct"]=True
    assert score_diagnosis([row],judge)["ACCreason"]==1
    judge["decisions"][0]["prediction_sha256"]="wrong"
    with pytest.raises(ValueError,match="does not match"):score_diagnosis([row],judge)


def test_public_benchmark_adapters(tmp_path):
    csv=tmp_path/"rup.csv"
    csv.write_text('Question,Question_typos,Answer\nWhat is 1+1?,Wat is 1+1?,2\n')
    prepare_rup(csv,tmp_path/"rup.jsonl","gsm8k","https://example.com/source","commit-a")
    rows,m=read_cache(tmp_path/"rup.jsonl","reliability")
    assert rows[0]["answer"]=="2" and m["provenance"]["revision"]=="commit-a"
    mr=tmp_path/"mr.json"
    mr.write_text(json.dumps([{"uuid":"a","question":"1+1?","model_output_steps":["1+1=3"],
        "model_output_solution_correctness":"wrong","model_output_solution_first_error_step":1,
        "model_output_solution_first_error_reason":"bad arithmetic"}]))
    prepare_mr(mr,tmp_path/"mr.jsonl","mr-gsm8k","https://example.com/source","commit-b")
    rows,_=read_cache(tmp_path/"mr.jsonl","diagnosis")
    assert rows[0]["solution"]=="1. 1+1=3" and not rows[0]["gold_correct"]


def test_real_hf_generation_path_and_checkpointing(config):
    # The ordinary HF generate path is exercised with the same tiny Qwen2, in
    # addition to the binary smoke sampler. No downloaded weights required.
    model,tok=load_model(config.with_changes(gradient_checkpointing=True))
    cfg=config.with_changes(tiny=False,completion_tokens=4,top_k=5)
    roll=Rollouts(tok,cfg)
    samples=roll.generate(model,tok.encode("agent question"),2,4,"real_hf",old=True)
    assert all(len(s.tokens)>0 and s.old_logps.shape[0]==len(s.tokens) for s in samples)
    from gasp.models import token_logps
    model.train()
    loss=-token_logps(model,samples[0].prompt,samples[0].tokens,cfg).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_reasoning_wrappers_do_not_become_guidance(config):
    _,tok=load_model(config)
    tokens=repair_target("internal clean polluted </think>1 + 1 = 2",tok)
    assert tok.decode(tokens)=="1 + 1 = 2"
    assert repair_target("<think>unfinished",tok)==[]
    window,tagged=pollution("I will edit </think><polluted>1 + 1 = 3</polluted>")
    assert tagged and window=="1 + 1 = 3"


def test_prompt_budget_preserves_complete_context_or_fails(config):
    class Words:
        def encode(self,text,**kw):return text.split()
        def decode(self,tokens,**kw):return " ".join(tokens)
    p=Prompts(Words(),config.with_changes(prompt_tokens=8))
    build=lambda prefix:f"QUESTION {prefix} POLLUTION"
    ids=p.fit(build, " ".join(["PREFIX"]*6))
    assert ids==["QUESTION"]+["PREFIX"]*6+["POLLUTION"]
    with pytest.raises(ValueError,match="refusing to truncate"):
        p.fit(build, " ".join(["PREFIX"]*20))
    with pytest.raises(ValueError,match="exceed"):
        p.fit(lambda prefix:" ".join(["QUESTION"]*10),"")
