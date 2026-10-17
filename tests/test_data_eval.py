import json
from pathlib import Path
import pytest

from gasp.data import read_cache, write_cache, assert_disjoint, question_hash
from gasp.evaluation import aggregate
from gasp.models import rng_state
from test_training import assert_tree_equal


def test_evaluation_cache_identity_and_rng_isolation(factory):
    u,s=factory("unified"),factory("split")
    assert u.eval_manifest["sha256"]==s.eval_manifest["sha256"]
    assert u.eval_rows==s.eval_rows
    before=rng_state()
    um=u.evaluate()
    assert_tree_equal(before,rng_state())
    sm=s.evaluate()
    assert um["metrics"]==sm["metrics"]
    up=(u.output/"eval/step-000000/predictions.jsonl").read_text()
    sp=(s.output/"eval/step-000000/predictions.jsonl").read_text()
    assert up==sp


def test_cache_tamper_and_overlap(tmp_path):
    rows=[{"id":"a","question":"Question A","split":"train"}]
    p=tmp_path/"train.jsonl"
    write_cache(p,rows,{"kind":"train"})
    with pytest.raises(FileExistsError):write_cache(p,rows,{"kind":"train"})
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="hash"):read_cache(p)
    with pytest.raises(ValueError,match="overlap"):
        assert_disjoint(rows,[{"id":"b","question":"  QUESTION   A ","split":"test"}])


def test_end_to_end_and_aggregation_guards(factory,tmp_path):
    u,s=factory("unified"),factory("split")
    # Re-seed each run via fresh factory so the conditions receive paired seeds.
    u=factory("unified");u.run()
    s=factory("split");s.run()
    report=aggregate([u.output,s.output],tmp_path/"summary",allow_pilot=True)
    assert len(report["per_seed"])==1
    assert report["summary"]["clean_full"]["unified"]["sample_std"] is None
    assert s.telemetry()["unique_role_parameter_bytes"]==2*u.telemetry()["unique_role_parameter_bytes"]
    assert s.telemetry()["optimizer_bytes"]==2*u.telemetry()["optimizer_bytes"]
    assert s.telemetry()["snapshot_bytes"]==0 and u.telemetry()["snapshot_bytes"]>0
    assert [h["lr"] for h in u.history]==[h["lr"] for h in s.history]
    assert s.roles.versions["agent"]==s.cfg.role_steps
    assert u.roles.versions["agent"]==2*u.cfg.role_steps
    with pytest.raises(ValueError):aggregate([u.output,s.output],tmp_path/"bad")
    path=s.output/"eval/step-000008/metrics.json"
    m=json.loads(path.read_text());m["eval_sha256"]="tampered"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="eval_sha256"):
        aggregate([u.output,s.output],tmp_path/"bad",allow_pilot=True)


def test_preparation_uses_real_initial_model_and_fixed_filter(config,tmp_path):
    from gasp.data import prepare_train
    train=[{"id":f"raw-{i}","question":f"train {i}","answer":str(i%2),
            "reference_trace":str(i%2),"split":"train"} for i in range(96)]
    evaluation=[{"id":f"held-{i}","question":f"test {i}","answer":str(i%2),
                 "reference_trace":str(i%2),"split":"test"} for i in range(4)]
    write_cache(tmp_path/"raw-train.jsonl",train,{"kind":"raw_train"})
    write_cache(tmp_path/"raw-eval.jsonl",evaluation,{"kind":"raw_eval"})
    prepare_train(config,tmp_path/"raw-train.jsonl",tmp_path/"train.jsonl",96)
    retained,tm=read_cache(tmp_path/"train.jsonl","train")
    assert retained and all(r["filter_scores"]==[1,1,1,1] for r in retained)
    assert tm["provenance"]["counts"]["train_filter"]["sequences"]==96*4
    assert all("initial_trace" not in r for r in retained)
    assert_disjoint(retained,evaluation)
